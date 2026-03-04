import math
import hashlib
import mido
import io
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Universal Prime Geometry Constants
PHI = (1 + math.sqrt(5)) / 2
GRID = 32

@dataclass
class ExpertNode:
    prime: int
    name: str
    category: str # 'dsl', 'hard', 'local'
    coord: tuple # (x, y, z)

class ARCUPALattice:
    """
    Universal Prime Architecture for ARC-AGI.
    Maps every solving 'expert' to a unique prime number in a 3D semantic lattice.
    """
    def __init__(self):
        self.experts: Dict[int, ExpertNode] = {}
        self.by_name: Dict[str, ExpertNode] = {}
        self._initialize_experts()

    def _initialize_experts(self):
        # First ~65 primes for our initial expert pool
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 
                  73, 79, 83, 89, 97, 101, 103, 107, 113, 127, 131, 137, 139, 149, 151, 157, 
                  163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 
                  241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317]
        
        # Categories and their experts
        dsl_primitives = ["identity", "rotate", "flip", "recolor", "crop", "scale", "outline", "fill"]
        hard_heuristics = ["recursive_tiling", "relational_graph", "nca_flood", "nca_outline"]
        local_attractors = ["local_rule_single", "local_rule_iterative"]

        pool = []
        for name in dsl_primitives: pool.append((name, 'dsl'))
        for name in hard_heuristics: pool.append((name, 'hard'))
        for name in local_attractors: pool.append((name, 'local'))

        for i, (name, cat) in enumerate(pool):
            p = primes[i]
            x = (p % 21) * GRID // 21
            y = (p % 369) * GRID // 369
            z = int(math.log(max(p, 2)) / math.log(PHI)) % GRID
            node = ExpertNode(p, name, cat, (x, y, z))
            self.experts[p] = node
            self.by_name[name] = node

    def get_expert_by_prime(self, p: int) -> Optional[ExpertNode]:
        return self.experts.get(p)

    def get_prime_by_name(self, name: str) -> Optional[int]:
        node = self.by_name.get(name)
        return node.prime if node else None

class ARCPrimeMidiRouter:
    """
    Encodes ARC expert activation patterns into deterministic MIDI signals.
    """
    def __init__(self, lattice: ARCUPALattice):
        self.lattice = lattice
        self.TICKS_PER_BEAT = 480

    def encode_routing(self, activations: Dict[int, float], task_id: str) -> bytes:
        """
        activations: {prime: strength}
        """
        mid = mido.MidiFile(type=1, ticks_per_beat=self.TICKS_PER_BEAT)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        # Meta info
        track.append(mido.MetaMessage('track_name', name=f"ARC_UPA:{task_id[:8]}", time=0))
        
        # Sort activations to ensure MIDI determinism
        for p in sorted(activations.keys()):
            strength = activations[p]
            node = self.lattice.get_expert_by_prime(p)
            if not node: continue
            
            # Simple MIDI note mapping: prime index -> note
            note = 21 + list(self.lattice.experts.keys()).index(p)
            velocity = int(strength * 127)
            
            track.append(mido.Message('note_on', note=note, velocity=velocity, time=0))
            track.append(mido.Message('note_off', note=note, velocity=0, time=480))
            
        buf = io.BytesIO()
        mid.save(file=buf)
        return buf.getvalue()

    def generate_seed(self, midi_bytes: bytes) -> int:
        h = hashlib.sha256(midi_bytes).hexdigest()
        return int(h, 16)

if __name__ == "__main__":
    lattice = ARCUPALattice()
    print(f"UPA Lattice Initialized with {len(lattice.experts)} experts.")
