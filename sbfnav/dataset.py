"""Strict CityNav/CityRefer loading without implicit split mixing."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import Levenshtein


VALID_SPLITS = ("train_seen", "val_seen", "val_unseen", "test_unseen")
VALID_DIFFICULTIES = (None, "easy", "medium", "hard")


def _pose4(entry: Sequence[float]) -> np.ndarray:
    if len(entry) == 5:
        x, y, z, yaw, _pitch = entry
    elif len(entry) == 6:
        x, y, z, dx, dy, _dz = entry
        yaw = float(np.arctan2(dy, dx))
    else:
        raise ValueError(f"expected a 5- or 6-element pose, got {len(entry)}")
    return np.asarray((x, y, z, yaw), dtype=np.float64)


@dataclass(frozen=True)
class Landmark:
    map_name: str
    object_id: int
    name: str
    position: np.ndarray
    contour: np.ndarray

    @property
    def centroid(self) -> np.ndarray:
        # Paper requests polygon centroids. Use the shoelace centroid and fall
        # back to the point mean for degenerate annotations.
        points = self.contour
        x, y = points[:, 0], points[:, 1]
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        area2 = cross.sum()
        if abs(area2) < 1e-9:
            return points.mean(axis=0)
        return np.asarray(
            (
                ((x + np.roll(x, -1)) * cross).sum() / (3.0 * area2),
                ((y + np.roll(y, -1)) * cross).sum() / (3.0 * area2),
            ),
            dtype=np.float64,
        )

    @property
    def area(self) -> float:
        x, y = self.contour[:, 0], self.contour[:, 1]
        return float(abs((x * np.roll(y, -1) - np.roll(x, -1) * y).sum()) / 2.0)


@dataclass(frozen=True)
class CityNavRecord:
    split: str
    index: int
    map_name: str
    object_id: int
    description_id: int
    instruction: str
    trajectory: np.ndarray
    target_position: np.ndarray
    marker_position: np.ndarray
    raw: Mapping[str, Any]

    @property
    def episode_id(self) -> tuple[str, int, int]:
        return self.map_name, self.object_id, self.description_id

    @property
    def start_pose(self) -> np.ndarray:
        return self.trajectory[0]


class CityReferCatalog:
    """CityRefer objects plus exact processed-description landmark links."""

    def __init__(self, root: str | Path):
        root = Path(root)
        self.root = root
        self.objects: Dict[str, Dict[int, Mapping[str, Any]]] = {
            map_name: {int(key): value for key, value in objects.items()}
            for map_name, objects in json.loads((root / "objects.json").read_text()).items()
        }
        self.processed: Dict[str, Dict[int, List[Mapping[str, Any]]]] = {
            map_name: {int(key): value for key, value in objects.items()}
            for map_name, objects in json.loads(
                (root / "processed_descriptions.json").read_text()
            ).items()
        }
        self._named = self._index_named_landmarks()

    @staticmethod
    def _landmark(value: Mapping[str, Any]) -> Landmark:
        return Landmark(
            map_name=str(value["map_name"]),
            object_id=int(value["id"]),
            name=str(value["name"]),
            position=np.asarray(value["position"], dtype=np.float64),
            contour=np.asarray(value["contour"], dtype=np.float64),
        )

    def _index_named_landmarks(self) -> Dict[str, Dict[str, Landmark]]:
        result: Dict[str, Dict[str, Landmark]] = {}
        for map_name, objects in self.objects.items():
            selected: Dict[str, Landmark] = {}
            for value in objects.values():
                if not value.get("name"):
                    continue
                landmark = self._landmark(value)
                previous = selected.get(landmark.name)
                if previous is None or landmark.area > previous.area:
                    selected[landmark.name] = landmark
            result[map_name] = selected
        return result

    def target_object(self, record: CityNavRecord) -> Mapping[str, Any]:
        return self.objects[record.map_name][record.object_id]

    def all_landmarks(self, map_name: str) -> List[Landmark]:
        return list(self._named[map_name].values())

    @property
    def all_named_landmark_names(self) -> tuple[str, ...]:
        return tuple(
            sorted({name for landmarks in self._named.values() for name in landmarks})
        )

    def referenced_landmark_names(self, record: CityNavRecord) -> List[str]:
        descriptions = self.processed[record.map_name][record.object_id]
        if record.description_id >= len(descriptions):
            raise IndexError(f"invalid description id for {record.episode_id}")
        names = descriptions[record.description_id].get("landmarks", [])
        return list(dict.fromkeys(str(name) for name in names))

    def referenced_landmarks(self, record: CityNavRecord) -> List[Landmark]:
        available = self._named[record.map_name]
        if not available:
            raise KeyError(f"map {record.map_name} has no named CityRefer landmarks")
        return [self.resolve_landmark(record.map_name, name) for name in self.referenced_landmark_names(record)]

    @staticmethod
    def _normalize_landmark_name(name: str) -> str:
        return "".join(character for character in name.lower() if character.isalnum())

    def resolve_landmark(self, map_name: str, query_name: str) -> Landmark:
        """Resolve local aliases using exact normalized then edit-distance match.

        The supplied processed descriptions contain aliases such as ``Rd`` vs
        ``Road`` and optional ``building`` suffixes. This reproduces the
        deterministic text-only fallback used by the existing HETT code.
        """
        available = self._named[map_name]
        if query_name in available:
            return available[query_name]
        normalized_query = self._normalize_landmark_name(query_name)
        normalized_exact = [
            landmark
            for name, landmark in available.items()
            if self._normalize_landmark_name(name) == normalized_query
        ]
        if normalized_exact:
            return sorted(normalized_exact, key=lambda item: (item.name, item.object_id))[0]
        return min(
            available.values(),
            key=lambda item: (
                Levenshtein.distance(
                    normalized_query, self._normalize_landmark_name(item.name)
                ),
                item.name,
                item.object_id,
            ),
        )

    def landmark_resolutions(self, record: CityNavRecord) -> List[tuple[str, str]]:
        return [
            (query, self.resolve_landmark(record.map_name, query).name)
            for query in self.referenced_landmark_names(record)
        ]


class CityNavDataset(Sequence[CityNavRecord]):
    """Map-style loader that requires explicit authorization for Test-Unseen."""

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        *,
        version: str = "processed_citynav",
        difficulty: str | None = None,
        allow_test_unseen: bool = False,
        max_records: int | None = None,
    ):
        if split not in VALID_SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {VALID_SPLITS}")
        if difficulty not in VALID_DIFFICULTIES:
            raise ValueError(f"unknown difficulty {difficulty!r}")
        if split == "train_seen" and difficulty is not None:
            raise ValueError("training annotations have no difficulty subsets")
        if split == "test_unseen" and not allow_test_unseen:
            raise PermissionError(
                "test_unseen is sealed during development; pass allow_test_unseen=True "
                "only from the frozen final-evaluation workflow"
            )
        self.data_root = Path(data_root)
        self.version = version
        self.split = split
        self.difficulty = difficulty
        suffix = "" if difficulty is None else f"_{difficulty}"
        self.annotation_path = self.data_root / version / f"citynav_{split}{suffix}.json"
        if not self.annotation_path.is_file():
            raise FileNotFoundError(self.annotation_path)
        raw_records = json.loads(self.annotation_path.read_text())
        if max_records is not None:
            if max_records < 0:
                raise ValueError("max_records must be nonnegative")
            raw_records = raw_records[:max_records]
        self._raw_records = raw_records
        self._records: list[CityNavRecord | None] = [None] * len(raw_records)

    def _convert(self, index: int, raw: Mapping[str, Any]) -> CityNavRecord:
        if str(raw["split"]) != self.split:
            raise ValueError(
                f"annotation split mismatch in {self.annotation_path}: {raw['split']!r}"
            )
        if not (raw["object_ids"] and raw["ann_ids"] and raw["descriptions"]):
            raise ValueError(f"empty episode identifiers at record {index}")
        trajectory = np.stack([_pose4(entry) for entry in raw["trajectory"]])
        return CityNavRecord(
            split=self.split,
            index=index,
            map_name=f"{raw['area']}_block_{raw['block']}",
            object_id=int(raw["object_ids"][0]),
            description_id=int(raw["ann_ids"][0]),
            instruction=str(raw["descriptions"][0]),
            trajectory=trajectory,
            target_position=np.asarray(raw["target_positions"][-1], dtype=np.float64),
            marker_position=np.asarray(raw["marker_positions"][-1], dtype=np.float64),
            raw=raw,
        )

    def __len__(self) -> int:
        return len(self._raw_records)

    def __getitem__(self, index: int) -> CityNavRecord:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        record = self._records[index]
        if record is None:
            record = self._convert(index, self._raw_records[index])
            self._records[index] = record
        return record

    @property
    def map_names(self) -> set[str]:
        return {f"{raw['area']}_block_{raw['block']}" for raw in self._raw_records}

    @property
    def episode_ids(self) -> set[tuple[str, int, int]]:
        return {
            (f"{raw['area']}_block_{raw['block']}", int(raw["object_ids"][0]), int(raw["ann_ids"][0]))
            for raw in self._raw_records
        }

    @property
    def episode_index(self) -> dict[tuple[str, int, int], int]:
        return {
            (f"{raw['area']}_block_{raw['block']}", int(raw["object_ids"][0]), int(raw["ann_ids"][0])): index
            for index, raw in enumerate(self._raw_records)
        }

    @property
    def instructions(self) -> tuple[str, ...]:
        return tuple(str(raw["descriptions"][0]) for raw in self._raw_records)

    def validate_against_cityrefer(
        self, catalog: CityReferCatalog, *, require_instruction_match: bool = False
    ) -> int:
        """Validate target/link integrity and return verbatim text mismatches.

        Split JSON is authoritative for instruction text. Both supplied data
        versions contain intentional/non-semantic text differences from the
        CityRefer object descriptions, while their object and annotation links
        remain valid.
        """
        instruction_mismatches = 0
        for record in self:
            target = catalog.target_object(record)
            if not np.allclose(target["position"], record.target_position, atol=1e-6):
                raise ValueError(f"target mismatch for {record.episode_id}")
            descriptions = target["descriptions"]
            if record.description_id >= len(descriptions):
                raise ValueError(f"description index mismatch for {record.episode_id}")
            if descriptions[record.description_id] != record.instruction:
                instruction_mismatches += 1
            if require_instruction_match and descriptions[record.description_id] != record.instruction:
                raise ValueError(f"instruction mismatch for {record.episode_id}")
            catalog.referenced_landmarks(record)
        return instruction_mismatches


def assert_development_split_isolation(datasets: Iterable[CityNavDataset]) -> None:
    """Reject accidental Test-Unseen loading and unseen-map overlap."""
    datasets = list(datasets)
    if any(dataset.split == "test_unseen" for dataset in datasets):
        raise ValueError("development datasets must not include test_unseen")
    by_split = {dataset.split: dataset for dataset in datasets}
    train_maps = by_split.get("train_seen", CityNavDataset.__new__(CityNavDataset))
    seen_maps = set() if not hasattr(train_maps, "_records") else train_maps.map_names
    if "val_seen" in by_split:
        seen_maps |= by_split["val_seen"].map_names
    if "val_unseen" in by_split and seen_maps & by_split["val_unseen"].map_names:
        raise ValueError("seen and val_unseen map sets overlap")


def difficulty_membership(
    dataset: CityNavDataset,
    difficulties: Sequence[str] = ("easy", "medium", "hard"),
    *,
    allow_test_unseen: bool = False,
) -> dict[tuple[str, int, int], str]:
    """Return an exact one-to-one difficulty partition for a full split.

    This lets the frozen Test-Unseen evaluator execute every trajectory once
    and aggregate all official difficulty rows inside that single authorized
    process.  It does not load Test-Unseen unless its caller has already
    supplied the explicit final-evaluation authorization.
    """
    if dataset.difficulty is not None:
        raise ValueError("difficulty membership requires the full split dataset")
    if len(dataset.episode_ids) != len(dataset):
        raise ValueError("full split contains duplicate episode identifiers")
    membership: dict[tuple[str, int, int], str] = {}
    for difficulty in difficulties:
        subset = CityNavDataset(
            dataset.data_root,
            dataset.split,
            version=dataset.version,
            difficulty=difficulty,
            allow_test_unseen=allow_test_unseen,
        )
        if len(subset.episode_ids) != len(subset):
            raise ValueError(f"{difficulty} subset contains duplicate episode identifiers")
        for episode_id in subset.episode_ids:
            if episode_id in membership:
                raise ValueError(f"difficulty subsets overlap at {episode_id}")
            membership[episode_id] = difficulty
    missing = dataset.episode_ids - membership.keys()
    extra = membership.keys() - dataset.episode_ids
    if missing or extra:
        raise ValueError(
            f"difficulty subsets do not partition {dataset.split}: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    return membership
