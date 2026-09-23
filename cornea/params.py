"""Input parameters, read from a JSON file."""
from dataclasses import dataclass, asdict
import json


@dataclass
class CorneaParams:
    R_v: float = 5.25   # radial extent of the posterior face
    R_a: float = 6.5    # anterior radius
    Q_a: float = -0.61  # anterior asphericity
    R_p: float = 5.5    # posterior radius
    Q_p: float = -0.72  # posterior asphericity
    h: float = 0.5      # central thickness
    Nx: int = 50        # nodes along the radius
    Ny: int = 6         # nodes through the thickness

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            data = json.load(f)
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown parameter(s) in {path}: {sorted(unknown)}")
        return cls(**data)

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
