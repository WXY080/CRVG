import contextlib
import copy
import importlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from analysis.common import paths
from analysis.replay_thresholds import replay_split
from crvg.controller.apply import score_evidence, merge_decisions
from crvg.controller.features import build_risk_examples
from crvg.utils.data import fingerprint, write_json, read_json, write_jsonl
from crvg.verification.apply_gate import apply_crop
from tools.merge_ece import merge
from tests.fixtures import A, B, C, case, evidence, controller


def cli(module_name, args):
    module=importlib.import_module(module_name)
    with patch.object(sys,"argv",[module_name,*map(str,args)]):
        with contextlib.redirect_stdout(io.StringIO()):
            module.main()


def create_cached_run(directory, dataset="refcoco_val"):
    p=paths(directory,dataset)
    row,record=case()
    row.update(dataset=dataset,split="val")
    base=copy.deepcopy({"meta":{"synthetic":True},"results":[row]})
    base["results"][0]["candidates"]=[{"bbox":A,"source":"greedy"},{"bbox":C,"source":"sampled"}]
    views={"meta":{"source_sha256":fingerprint(base),"gamma0":.5},
           "records":{"0":{"routed":True,"transformed_candidates":[]}}}
    expanded=merge(base,views)
    crops={"meta":{"source_sha256":fingerprint(expanded)},
           "picks":[{"dataset_index":0,"status":"scored","current_probability":.8,
                     "candidates":[{"bbox":A,"p_yes":.8},{"bbox":C,"p_yes":.2}]}]}
    qwen=apply_crop(expanded,crops)
    ev,ps=evidence([copy.deepcopy(qwen["results"][0])],[record])
    ev["results"][0]["candidates"].append(copy.deepcopy(row["candidates"][-1]))
    ev["meta"]["source_sha256"]=fingerprint(qwen)
    ps["meta"]["source_sha256"]=fingerprint(ev)
    controller_dir=Path(directory)/"controller"
    controller(controller_dir)
    decisions,config,_=score_evidence(ev,ps,controller_dir)
    final=merge_decisions(qwen,ev,decisions)
    risk={"decisions":decisions,"source_sha256":fingerprint(qwen),"evidence_sha256":fingerprint(ev)}
    for key,value in {"base":base,"expanded":expanded,"crops":crops,"qwen":qwen,
                      "dino":ev,"pairs":ps,"final":final,"risk":risk}.items():
        write_json(p[key],value)
    return p


class WorkflowTests(unittest.TestCase):
    def test_controller_rejects_incompatible_scoring_protocol(self):
        from crvg.controller.apply import load_controller
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            controller(root/"original")
            cfg=read_json(root/"original/risk_controller_config.json")
            cfg["scoring_backend"]="different_protocol"
            write_json(root/"original/risk_controller_config.json",cfg)
            with self.assertRaisesRegex(ValueError,"scoring backend"):
                load_controller(root/"original")

    def test_training_serialization_cycle(self):
        train,val=[],[]
        for domain_idx,domain in enumerate(("refcoco","refcoco+","refcocog")):
            for offset,output in ((0,train),(100,val)):
                for role_idx,role in enumerate(("rescue","protect","neutral")):
                    row,record=case(domain_idx*10+role_idx+offset,domain,role)
                    data,picks=evidence([row],[record])
                    output.extend(build_risk_examples(data,picks,"synthetic",require_train=True)[0])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write_jsonl(root/"train.jsonl",train)
            write_jsonl(root/"cal.jsonl",val)
            cli("crvg.controller.train",["--train",root/"train.jsonl","--val",root/"cal.jsonl",
                "--output-dir",root/"trained","--epochs","2","--batch-size","9",
                "--gates","0,10","--damage-costs","2","--abstain-costs","0.25","--device","cpu","--no-strict"])
            cfg=read_json(root/"trained/risk_controller_config.json")
            self.assertTrue(cfg["selected_policy"]["require_permutation_agree"])
            self.assertNotIn("projected_full_delta_acc50",cfg)
            self.assertTrue((root/"trained/risk_controller.pt").is_file())

    def test_cached_analysis_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            create_cached_run(root)
            for module in ("tools.show_results","analysis.rescue_damage","analysis.routing_efficiency"):
                cli(module,["--log-dir",root,"--datasets","refcoco_val"])
            selected,report=replay_split(root,"refcoco_val")
            self.assertEqual(report["delta_pp"]["acc0.5"],0)
            self.assertEqual(selected["results"][0]["pred_bbox"],B)
            cli("analysis.replay_thresholds",["--log-dir",root,"--datasets","refcoco_val",
                                             "--output-dir",root/"replay"])
            cli("analysis.threshold_sweep",["--log-dir",root,"--datasets","refcoco_val",
                "--output-dir",root/"sweep","--gamma0-grid","0.5","--gate-grid","0.3","--gamma1-grid","0.35"])
            self.assertIn("Average",(root/"results_table.md").read_text())
            self.assertTrue((root/"sweep/threshold_sensitivity.pdf").is_file())
            anatomy=read_json(root/"rescue_damage.json")["refcoco_val"]
            self.assertEqual(anatomy["pairwise_zero_gate"]["rescue"],1)

    def test_resume_manifest_checks_output_content(self):
        from crvg.pipeline import run_stage
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/"source.json"
            output=root/"out.json"
            write_json(source,{"v":1})
            def run(*a,**kw):
                write_json(output,{"complete":True})
            with patch("crvg.pipeline.subprocess.run",side_effect=run) as mock:
                with contextlib.redirect_stdout(io.StringIO()):
                    run_stage("fake.module",[],[source],[output],sys.executable)
                    run_stage("fake.module",[],[source],[output],sys.executable)
                    self.assertEqual(mock.call_count,1)
                    write_json(output,{"corrupted":True})
                    run_stage("fake.module",[],[source],[output],sys.executable)
                    self.assertEqual(mock.call_count,2)

    def test_cli_help_does_not_load_models(self):
        modules=["crvg.pipeline","crvg.candidate_generation.ece","crvg.verification.crop_verifier",
                 "crvg.verification.dino_detector","crvg.verification.pairwise","crvg.verification.apply_gate",
                 "crvg.controller.build_data","crvg.controller.train","crvg.controller.apply",
                 "tools.merge_ece","tools.import_candidates","tools.prepare_data",
                 "tools.visualize","tools.show_results","analysis.replay_thresholds","analysis.threshold_sweep",
                 "analysis.routing_efficiency","analysis.rescue_damage"]
        for module in modules:
            with self.subTest(module=module):
                with self.assertRaises(SystemExit) as exit:
                    cli(module,["--help"])
                self.assertEqual(exit.exception.code,0)


if __name__ == "__main__":
    unittest.main()
