from dataclasses import dataclass


@dataclass
class DataConfigArtifact:
    org_train_dirr:str
    org_valid_dirr:str
    org_test_dirr: str
    gt_train_dirr: str
    gt_valid_dirr: str
    gt_test_dirr: str