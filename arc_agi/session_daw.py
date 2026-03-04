import os
import json
import hashlib
import mido
import io
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
from arc_types import ARCTask, Grid
from arc_bra import EigenCharge, task_charge, arc_task_resonance

TAPE_DIR = os.path.expanduser("~/.arc_agi/tapes")

@dataclass
class ExpertTape:
    task_id: str
    charge: Dict[str, int] # hash, trace, det
    midi_bytes_hex: str    # Hex encoded MIDI data
    solve_path: List[str]  # The successful expert sequence
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class SessionDAW:
    """
    Session Digital Audio Workstation for ARC-AGI.
    Records successful solve paths as 'Expert Tapes' and replays them
    for resonant tasks.
    """
    def __init__(self, tape_dir: str = TAPE_DIR, ped_master_dir: Optional[str] = None):
        self.tape_dir = tape_dir
        self.ped_master_dir = ped_master_dir or os.path.expanduser("~/.arc_agi/pedagogy/master_tapes")
        os.makedirs(self.tape_dir, exist_ok=True)
        os.makedirs(self.ped_master_dir, exist_ok=True)
        self.tapes: List[ExpertTape] = self._load_tapes(self.tape_dir)
        self.master_tapes: List[ExpertTape] = self._load_tapes(self.ped_master_dir)

    def _load_tapes(self, directory: str) -> List[ExpertTape]:
        tapes = []
        if not os.path.exists(directory):
            return tapes
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(directory, filename), 'r') as f:
                        data = json.load(f)
                        tapes.append(ExpertTape(**data))
                except Exception as e:
                    print(f"Error loading tape {os.path.join(directory, filename)}: {e}")
        return tapes

    def record(self, task: ARCTask, midi_bytes: bytes, solve_path: List[str], score: float = 1.0):
        """
        Record a successful session onto a persistent Expert Tape.
        """
        charge = task_charge(task)
        tape = ExpertTape(
            task_id = task.task_id,
            charge = {"hash": charge.hash, "trace": charge.trace, "det": charge.det},
            midi_bytes_hex = midi_bytes.hex(),
            solve_path = solve_path,
            score = score
        )
        
        # Persistence
        tape_path = os.path.join(self.tape_dir, f"{task.task_id}.json")
        with open(tape_path, 'w') as f:
            json.dump(asdict(tape), f, indent=2)
        
        # Update in-memory index
        # Replace if better score or same task
        for i, existing in enumerate(self.tapes):
            if existing.task_id == task.task_id:
                self.tapes[i] = tape
                break
        else:
            self.tapes.append(tape)

    def get_master_tape(self, name: str) -> Optional[ExpertTape]:
        """
        Retrieve a Master Tape by name from the pedagogical collection.
        """
        for tape in self.master_tapes:
            if tape.task_id == name:
                return tape
        return None

    def playback(self, task: ARCTask, min_resonance: int = 1) -> Optional[ExpertTape]:
        """
        Find a resonant tape and return the expert sequence for replay.
        """
        charge = task_charge(task)
        best_tape = None
        best_res = -1
        
        for tape in self.tapes:
            tape_charge = EigenCharge(
                hash=tape.charge["hash"], 
                trace=tape.charge["trace"], 
                det=tape.charge["det"]
            )
            res = arc_task_resonance(charge, tape_charge)
            if res >= min_resonance:
                if res > best_res:
                    best_res = res
                    best_tape = tape
                elif res == best_res and (best_tape is None or tape.score > best_tape.score):
                    best_tape = tape
                    
        return best_tape

if __name__ == "__main__":
    daw = SessionDAW()
    print(f"Session DAW initialized with {len(daw.tapes)} tapes.")
