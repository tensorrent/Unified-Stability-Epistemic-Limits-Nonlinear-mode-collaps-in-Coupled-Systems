from __future__ import annotations
from arc_types import ARCTask, grid_shape, grid_unique_colors, background_color, extract_objects
from arc_bra import upg_ordered_primitives

class ILMDeterministicEngine:
    """
    Intelligent Language Model (ILM) - Deterministic Engine for ARC AGI.
    Replaces stochastic LLM generation with an exact geographical, chronological, 
    and ontological LAC audit.
    """
    
    @staticmethod
    def analyze(task: ARCTask) -> str:
        """
        Produce a deterministic LAC audit trace for the task.
        Returns a string containing the analysis report.
        """
        _, audit_log = ILMDeterministicEngine._synthesize_with_audit(task)
        return "\n".join(audit_log)

    @staticmethod
    def synthesize(task: ARCTask) -> list[str]:
        """
        Produce candidate python transform functions by computing deterministic 
        geographical constraints.
        Returns a list of python code string candidates.
        """
        candidates, _ = ILMDeterministicEngine._synthesize_with_audit(task)
        return candidates

    @staticmethod
    def _synthesize_with_audit(task: ARCTask) -> tuple[list[str], list[str]]:
        """Internal helper to compute both candidates and audit log."""
        candidates = []
        audit_log = []
        
        audit_log.append("\n" + "="*50)
        audit_log.append("  [LAC AUDIT] ILM DETERMINISTIC ENGINE TRACE")
        audit_log.append("="*50)
        
        if not task.train:
            return candidates

        # --- WHAT: Ontology ---
        in_grid = task.train[0].input
        bg = background_color(in_grid)
        colors = sorted(list(grid_unique_colors(in_grid)))
        objects = extract_objects(in_grid, bg=bg)
        
        audit_log.append(f"→ WHAT: Input grid contains {len(objects)} objects. Colors: {colors} (bg: {bg})")
        
        # --- WHERE: Geography ---
        shape_differs = False
        scale_x, scale_y = 1, 1
        for pair in task.train:
            if not pair.output: continue
            ih, iw = grid_shape(pair.input)
            oh, ow = grid_shape(pair.output)
            if (ih, iw) != (oh, ow):
                shape_differs = True
                scale_x = ow // max(iw, 1)
                scale_y = oh // max(ih, 1)
                break
                
        if shape_differs:
            audit_log.append(f"→ WHERE: Output shape differs from input. Derived integer scale geometry: x{scale_x}, y{scale_y}")
        else:
            audit_log.append("→ WHERE: Coordinate geometry invariant (shape identical).")
            
        # --- WHEN / WHY: Ontology Delta ---
        color_remap = False
        in_vocab = set()
        out_vocab = set()
        for pair in task.train:
            if not pair.output: continue
            in_colors = grid_unique_colors(pair.input)
            out_colors = grid_unique_colors(pair.output)
            in_vocab.update(in_colors)
            out_vocab.update(out_colors)
            if out_colors - in_colors:
                color_remap = True
                
        if color_remap:
            audit_log.append("→ WHEN: Geometric color ontology changed. Suggests flood fill or recolor operations.")
        else:
            audit_log.append("→ WHEN: Color ontology invariant. Pure spatial transformation isolated.")
            
        audit_log.append("→ WHY: Reconciling differentials via UPG Geometric Router...")

        # --- HOW: Construct Sequence ---
        top_prims = upg_ordered_primitives(task, max_primitives=4)
        audit_log.append(f"→ HOW: Synthesizing AST from UPG Active Nodes: {', '.join(top_prims)}")
        
        # --- PHASE 7: Chain of Mathematics (CoM) Integration ---
        try:
            from arc_com import CoMEngine
            com = CoMEngine()
            if task.train and task.train[0].output:
                com_data = com.synthesize_sequence(task.train[0].input, task.train[0].output)
                audit_log.append(f"→ CoM REASONER: Deduced metric {com_data['target_coords']}. Path: {com_data['sequence']}")
                com_affine = com_data['tensors']['affine']
                com_color = com_data['tensors']['color']
                
                # --- PHASE 9: Neuromorphic Topology Routing ---
                from arc_neuro import NeuromorphicBrain
                brain = NeuromorphicBrain(spike_threshold=0.5)
                # Phase 15 & 16: Memory Boost stimulus from DAW Recall & Pedagogy
                recall = task.metadata.get("daw_recall")
                ped_recall = task.metadata.get("pedagogical_recall")
                spiked_top_prims = brain.route_ast(task.train[0].input, task.train[0].output, 
                                                  com_data, memory_boost=recall,
                                                  pedagogical_boost=ped_recall)
                
                if spiked_top_prims:
                     audit_log.append(f"→ SNN BRAIN: Neuromorphic routing triggered {len(spiked_top_prims)} ast sequence action potentials.")
                     # We combine the biological spike sequence with upg bases
                     top_prims = spiked_top_prims + top_prims
                elif com_data.get('sequence'):
                     # Fallback to pure CoM if the action potential was too low
                     top_prims = com_data['sequence'] + top_prims
            else:
                com_affine = [[1, 0], [0, 1]]
                com_color = [[i if i != bg else bg for i in range(10)] for _ in range(10)]
        except Exception as e:
            com_affine = [[1, 0], [0, 1]]
            com_color = [[i if i != bg else bg for i in range(10)] for _ in range(10)]
            audit_log.append(f"→ SNN / CoM REASONER: Execution failed - {e}")

        # --- PHASE 8: Abstract Algebra Constraint Resolving ---
        try:
            from arc_algebra import AlgebraEngine
            alg_engine = AlgebraEngine()
            if task.train and task.train[0].output:
                alg_cands = alg_engine.synthesize_algebraic_ast(task.train[0].input, task.train[0].output)
                if alg_cands:
                    audit_log.append(f"→ ALGEBRA SOLVER: Z3 Successfully mapped Diophantine constraints. Generated {len(alg_cands)} mathematically exact AST sequences.")
                    candidates.extend(alg_cands)
                else:
                    audit_log.append("→ ALGEBRA SOLVER: No viable Z3 metric overlap. Defaulting to heuristic sequence generation.")
        except Exception as e:
            audit_log.append(f"→ ALGEBRA SOLVER: Internal solver exception - {e}")
            
        # Candidate 1: Geographic Base (Scale/Crop)
        code1 = [
            "def transform(grid):",
            "    # [LAC AUDIT: Geographic Base]",
            "    import copy",
            "    out = copy.deepcopy(grid)",
        ]
        if shape_differs:
            if scale_x == 2 and scale_y == 2:
                code1.extend(["    from arc_types import upscale", "    out = upscale(out, 2)"])
            elif scale_x == 3 and scale_y == 3:
                code1.extend(["    from arc_types import upscale", "    out = upscale(out, 3)"])
            elif scale_x < 1 and scale_y < 1:
                code1.extend(["    from arc_types import crop_to_content", f"    out = crop_to_content(out, {bg})"])
        code1.append("    return out")
        candidates.append("\n".join(code1))
        
        # Parameter Extraction Heuristics (Phase 3)
        # Deduce dynamic background
        bg_out = bg
        for pair in task.train:
            if pair.output:
                out_bg = background_color(pair.output)
                if out_bg != bg:
                    bg_out = out_bg
                    break
        
        # Deduce divider color for splits
        split_color = bg
        ih = len(in_grid)
        iw = len(in_grid[0]) if ih > 0 else 0
        for c in colors:
            if c != bg:
                cnt = sum(row.count(c) for row in in_grid)
                if cnt == ih or cnt == iw or cnt == ih * 2 or cnt == iw * 2:
                    split_color = c
                    break
        
        # Phase 4: Z-Axis Object Iteration Threshold
        from arc_bra import task_upg_coord
        vx, vy, vz = task_upg_coord(task)
        is_iterative_topology = (vz > 20)
        
        # Generate combination candidates based on UPG primitives
        # Candidate 2: Chain of Top 1 and Top 2 primitives
        for i, prim in enumerate(top_prims):
            if is_iterative_topology:
                code = [
                    "def transform(grid):",
                    f"    # [LAC AUDIT: UPG Node {prim} (Iterative Toplogy: Z>20)]",
                    "    import copy",
                    "    from arc_types import extract_objects",
                    "    from arc_dsl_ext import place_object",
                    f"    out = copy.deepcopy(grid)",
                    f"    objects = extract_objects(grid, bg={bg})",
                    "    for obj in objects:",
                    "        obj_grid = obj['grid']",
                ]
            else:
                code = [
                    "def transform(grid):",
                    f"    # [LAC AUDIT: UPG Node {prim}]",
                    "    import copy",
                    "    out = copy.deepcopy(grid)",
                ]
            
            # Helper to inject a primitive gracefully
            def inject_prim(p_name, c_list, is_loop=False):
                target = "obj_grid" if is_loop else "out"
                indent = "        " if is_loop else "    "
                
                if p_name == "recolor" or p_name == "replace_colors":
                    fc_list = sorted(list(in_vocab - out_vocab))
                    tc_list = sorted(list(out_vocab - in_vocab))
                    if len(fc_list) == 1 and len(tc_list) == 1:
                        c_list.extend([f"{indent}from arc_types import recolor", f"{indent}{target} = recolor({target}, {fc_list[0]}, {tc_list[0]})"])
                        audit_log.append(f"   ↳ Ast injected: recolor({target}, {fc_list[0]}, {tc_list[0]})")
                    elif fc_list and tc_list:
                        m_str = "{" + ", ".join(f"{f}: {t}" for f, t in zip(fc_list, tc_list)) + "}"
                        c_list.extend([f"{indent}from arc_types import replace_colors", f"{indent}{target} = replace_colors({target}, {m_str})"])
                        audit_log.append(f"   ↳ Ast injected: replace_colors({target}, {m_str})")
                    else:
                        c_list.extend([f"{indent}from arc_types import recolor", f"{indent}{target} = recolor({target}, {colors[0] if colors else 1}, {colors[-1] if colors else 2})"])
                        
                elif p_name in ["rot90", "rot180", "rot270", "reflect_h", "reflect_v", "reflect_diag", "reflect_anti"]:
                    c_list.extend([f"{indent}from arc_types import {p_name}", f"{indent}{target} = {p_name}({target})"])
                    audit_log.append(f"   ↳ Ast injected: {p_name}({target})")
                    
                elif p_name in ["fill_holes", "largest_object", "smallest_object", "border_objects", "interior_objects"]:
                    c_list.extend([f"{indent}from arc_dsl_ext import {p_name}", f"{indent}{target} = {p_name}({target}, {bg})"])
                    audit_log.append(f"   ↳ Ast injected: {p_name}({target}, {bg})")
                    
                elif p_name in ["outline", "dilate", "erode", "convex_hull_fill"]:
                    c_list.extend([f"{indent}from arc_dsl_ext import {p_name}", f"{indent}{target} = {p_name}({target}, bg={bg_out})"])
                    audit_log.append(f"   ↳ Ast injected: {p_name}({target}, bg={bg_out})")
                    
                elif p_name == "convex_hull":
                    c_list.extend([f"{indent}from arc_dsl_ext import convex_hull_fill", f"{indent}{target} = convex_hull_fill({target}, bg={bg_out})"])
                    
                elif p_name in ["crop_to_content"]:
                    c_list.extend([f"{indent}from arc_types import crop_to_content", f"{indent}{target} = crop_to_content({target}, {bg})"])
                    audit_log.append(f"   ↳ Ast injected: crop_to_content({target}, {bg})")
                    
                elif p_name.startswith("gravity_"):
                    d = p_name.split("_")[1]
                    c_list.extend([f"{indent}from arc_dsl_ext import gravity", f"{indent}{target} = gravity({target}, '{d}', {bg})"])
                    audit_log.append(f"   ↳ Ast injected: gravity({target}, '{d}', {bg})")
                    
                elif p_name == "split_divider":
                    c_list.extend([f"{indent}from arc_dsl_ext import split_by_divider", f"{indent}{target} = split_by_divider({target}, {split_color})[0]"])
                    audit_log.append(f"   ↳ Ast injected: split_by_divider({target}, {split_color})[0]")
                    
                elif p_name == "upscale_2":
                    c_list.extend([f"{indent}from arc_types import upscale", f"{indent}{target} = upscale({target}, 2)"])
                elif p_name == "downscale_2":
                    c_list.extend([f"{indent}from arc_types import downscale", f"{indent}{target} = downscale({target}, 2)"])
                elif p_name == "upscale_3":
                    c_list.extend([f"{indent}from arc_types import upscale", f"{indent}{target} = upscale({target}, 3)"])
                    
                elif p_name == "hstack_self":
                    c_list.extend([f"{indent}from arc_types import hstack", f"{indent}{target} = hstack({target}, {target})"])
                elif p_name == "vstack_self":
                    c_list.extend([f"{indent}from arc_types import vstack", f"{indent}{target} = vstack({target}, {target})"])
                elif p_name == "tile_2x2":
                    c_list.extend([f"{indent}from arc_types import tile", f"{indent}{target} = tile({target}, 2, 2)"])
                    
                elif p_name == "complete_h_sym":
                    c_list.extend([f"{indent}from arc_dsl_ext import complete_h_symmetry", f"{indent}{target} = complete_h_symmetry({target}, {bg})"])
                elif p_name == "complete_v_sym":
                    c_list.extend([f"{indent}from arc_dsl_ext import complete_v_symmetry", f"{indent}{target} = complete_v_symmetry({target}, {bg})"])
                elif p_name == "complete_rot180":
                    c_list.extend([f"{indent}from arc_dsl_ext import complete_rot180_symmetry", f"{indent}{target} = complete_rot180_symmetry({target}, {bg})"])
                elif p_name == "enforce_h_sym":
                    c_list.extend([f"{indent}from arc_dsl_ext import enforce_h_symmetry", f"{indent}{target} = enforce_h_symmetry({target})"])
                elif p_name == "enforce_v_sym":
                    c_list.extend([f"{indent}from arc_dsl_ext import enforce_v_symmetry", f"{indent}{target} = enforce_v_symmetry({target})"])
                    
                elif p_name == "extract_bounding_box":
                    c_list.extend([f"{indent}from arc_dsl_ext import extract_bounding_box", f"{indent}{target} = extract_bounding_box({target}, {bg})"])
                    
                elif p_name == "flood_fill":
                    c_list.extend([f"{indent}from arc_dsl_ext import flood_fill", f"{indent}{target} = flood_fill({target}, 0, 0, {bg}, {bg_out if bg_out != bg else 1})"])
                    
                elif p_name == "find_and_replace_pattern":
                    # For now just swapping identical reference targets to prevent crash if evaluated
                    c_list.extend([f"{indent}from arc_dsl_ext import find_and_replace_pattern", f"{indent}{target} = find_and_replace_pattern({target}, {target}, {target}, {bg})"])
                    
                elif p_name == "draw_line_until":
                    c_list.extend([f"{indent}from arc_dsl_ext import draw_line_until", f"{indent}{target} = draw_line_until({target}, 0, 0, 'right', 1)"])
                    
                elif p_name == "einsum_affine":
                    c_list.extend([f"{indent}from arc_dsl_ext import einsum_affine", f"{indent}{target} = einsum_affine({target}, {com_affine})"])
                    
                elif p_name == "einsum_color_map":
                    c_list.extend([f"{indent}from arc_dsl_ext import einsum_color_map", f"{indent}c_mat = {com_color}", f"{indent}{target} = einsum_color_map({target}, c_mat)"])
                
                elif p_name.startswith("kernel_"):
                    # Phase 16: High-Performance Rust Kernel Primitive
                    from arc_types import ARCTask
                    k_fn_map = {
                        "kernel_local_3x3": "solve_by_local_rules(task, wide=False)",
                        "kernel_local_5x5": "solve_by_local_rules(task, wide=True)",
                        "kernel_delta":     "solve_by_delta(task)",
                        "kernel_period":    "solve_by_period_v2(task)"
                    }
                    if p_name in k_fn_map:
                        k_fn = k_fn_map[p_name]
                        c_list.extend([f"{indent}from arc_kernel import {k_fn.split('(')[0]}", f"{indent}# Bridge to Rust Kernel: {p_name}", f"{indent}{target} = {k_fn}"])
                        audit_log.append(f"   ↳ Ast injected: {p_name} -> Rust Kernel")
                    
            inject_prim(prim, code, is_loop=is_iterative_topology)
            
            # Chain the next primitive to build a 2-step pipeline if requested
            if i + 1 < len(top_prims):
                next_p = top_prims[i+1]
                inject_prim(next_p, code, is_loop=is_iterative_topology)
                
            if is_iterative_topology:
                code.extend([
                    "        out = place_object(out, obj_grid, obj['bbox'][0], obj['bbox'][1], bg=0)",
                ])
                
            code.append("    return out")
            candidates.append("\n".join(code))
        
        audit_log.append("="*50)
        return candidates, audit_log
