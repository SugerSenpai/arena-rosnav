from __future__ import annotations

import dataclasses
import os
from typing import (
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    overload,
)

import enum

import numpy as np
import scipy


class Namespace(str):
    def __call__(self, *args: str) -> Namespace:
        return Namespace(os.path.join(self, *args))

    @property
    def simulation_ns(self) -> Namespace:
        return Namespace(self.split("/")[0])

    @property
    def robot_ns(self) -> Namespace:
        return Namespace(self.split("/")[1])


EMPTY_LOADER = lambda *_, **__: Model(
    type=ModelType.UNKNOWN, name="", description="", path=""
)


class ModelType(enum.Enum):
    UNKNOWN = ""
    URDF = "urdf"
    SDF = "sdf"
    YAML = "yaml"


@dataclasses.dataclass(frozen=True)
class Model:
    type: ModelType
    name: str
    description: str
    path: str

    @property
    def mapper(self) -> Callable[[Model], Model]:
        """
        Returns a (Model)->Model mapper that simply returns this model
        """
        return lambda m: self

    def replace(self, **kwargs) -> Model:
        """
        Wrapper for dataclasses.replace
        **kwargs: properties to replace
        """
        return dataclasses.replace(self, **kwargs)


ForbiddenZone = Tuple[float, float, float]

PositionOrientation = Tuple[float, float, float]
Waypoint = PositionOrientation
Position = Tuple[float, float]


class ModelWrapper:
    _get: Callable[[Collection[ModelType]], Model]
    _name: str
    _override: Dict[ModelType, Tuple[bool, Callable[..., Model]]]

    def __init__(self, name: str):
        """
        Create new ModelWrapper
        @name: Name of the ModelWrapper (should match the underlying Models)
        """
        self._name = name
        self._override = dict()

    def clone(self) -> ModelWrapper:
        """
        Clone (shallow copy) this ModelWrapper instance
        """
        clone = ModelWrapper(self.name)
        clone._get = self._get
        clone._override = self._override
        return clone

    def override(
        self,
        model_type: ModelType,
        override: Callable[[Model], Model],
        noload: bool = False,
        name: Optional[str] = None,
    ) -> ModelWrapper:
        """
        Create new ModelWrapper with an overridden ModelType callback
        @model_type: Which ModelType to override
        @override: Mapping function (Model)->Model which replaces the loaded model with a new one
        @noload: (default: False) If True, indicates that the original Model is not used by the override function and a dummy Model can be passed to it instead
        @name: (optional) If set, overrides name of ModelWrapper
        """
        clone = self.clone()
        clone._override = {**self._override, model_type: (noload, override)}

        if name is not None:
            clone._name = name

        return clone

    @overload
    def get(self, only: ModelType, **kwargs) -> Model:
        ...

    """
        load specific model
        @only: single accepted ModelType
    """

    @overload
    def get(self, only: Collection[ModelType], **kwargs) -> Model:
        ...

    """
        load specific model from collection
        @only: collection of acceptable ModelTypes
    """

    @overload
    def get(self, **kwargs) -> Model:
        ...

    """
        load any available model
    """

    def get(
        self, only: ModelType | Collection[ModelType] | None = None, **kwargs
    ) -> Model:
        if only is None:
            only = self._override.keys()

        if isinstance(only, ModelType):
            return self.get([only])

        for model_type in only:
            if model_type in self._override:
                noload, mapper = self._override[model_type]

                if noload == True:
                    return mapper(EMPTY_LOADER())

                return mapper(self._get([model_type], **kwargs), **kwargs)

        return self._get(only, **kwargs)

    @property
    def name(self) -> str:
        """
        get name
        """
        return self._name

    @staticmethod
    def bind(
        name: str, callback: Callable[[Collection[ModelType]], Model]
    ) -> ModelWrapper:
        """
        Create new ModelWrapper with bound callback method
        """
        wrap = ModelWrapper(name)
        wrap._get = callback
        return wrap

    @staticmethod
    def Constant(name: str, models: Dict[ModelType, Model]) -> ModelWrapper:
        """
        Create new ModelWrapper from a dict of already existing models
        @name: name of model
        @models: dictionary of ModelType->Model mappings
        """

        def get(only: Collection[ModelType]) -> Model:
            if not len(only):
                only = list(models.keys())

            for model_type in only:
                if model_type in models:
                    return models[model_type]
            else:
                raise LookupError(
                    f"no matching model found for {name} (available: {list(models.keys())}, requested: {list(only)})"
                )

        return ModelWrapper.bind(name, get)

    @staticmethod
    def from_model(model: Model) -> ModelWrapper:
        """
        Create new ModelWrapper containing a single existing Model
        @model: Model to wrap
        """
        return ModelWrapper.Constant(name=model.name, models={model.type: model})


@dataclasses.dataclass(frozen=True)
class EntityProps:
    position: PositionOrientation
    name: str
    model: ModelWrapper
    extra: Dict


@dataclasses.dataclass(frozen=True)
class ObstacleProps(EntityProps):
    ...


@dataclasses.dataclass(frozen=True)
class DynamicObstacleProps(ObstacleProps):
    waypoints: List[Waypoint]


@dataclasses.dataclass(frozen=True)
class RobotProps(EntityProps):
    planner: str
    agent: str
    record_data: bool


def parse_Point3D(obj: Sequence, fill: float = 0.0) -> Tuple[float, float, float]:
    position: Tuple[float, ...] = tuple([float(v) for v in obj[:3]])

    if len(position) < 3:
        position = (*position, *((3 - len(position)) * [fill]))

    return (position[0], position[1], position[2])


@dataclasses.dataclass(frozen=True)
class Obstacle(ObstacleProps):
    @staticmethod
    def parse(obj: Dict, model: ModelWrapper) -> "Obstacle":
        name = str(obj.get("name", ""))
        position = parse_Point3D(obj.get("pos", (0, 0, 0)))

        return Obstacle(name=name, position=position, model=model, extra=obj)


@dataclasses.dataclass(frozen=True)
class DynamicObstacle(DynamicObstacleProps):
    waypoints: Iterable[Waypoint]

    @staticmethod
    def parse(obj: Dict, model: ModelWrapper) -> "DynamicObstacle":
        name = str(obj.get("name", ""))
        position = parse_Point3D(obj.get("pos", (0, 0, 0)))
        waypoints = [parse_Point3D(waypoint)
                     for waypoint in obj.get("waypoints", [])]

        return DynamicObstacle(
            name=name, position=position, model=model, waypoints=waypoints, extra=obj
        )

def _gen_init_pos(steps:int, x:int=1, y:int=0):
    steps = max(steps,1)

    while True:
        yield (1,1,0)
        
    while True:
        x += y==steps
        y %= steps
        yield (-x,y,0)
        y += 1

gen_init_pos = _gen_init_pos(10)

@dataclasses.dataclass(frozen=True)
class Robot(RobotProps):
    @staticmethod
    def parse(obj: Dict, model: ModelWrapper) -> "Robot":
        name = str(obj.get("name", ""))
        position = parse_Point3D(obj.get("pos", next(gen_init_pos)))
        planner = str(obj.get("planner",""))
        agent = str(obj.get("agent",""))
        record_data = bool(obj.get("record_data",False))

        return Robot(
            name=name,
            position=position,
            planner=planner,
            model=model,
            agent=agent,
            record_data=record_data,
            extra=obj,
        )











_WorldWall = Tuple[Tuple[int, int], Tuple[int, int]]
WorldWalls = Collection[_WorldWall]
WorldObstacles = Collection[Obstacle]

class _WallLines(Dict[float, List[Tuple[float, float]]]):
    """
    Helper class for efficiently merging collinear line segments
    """

    _inverted: bool

    def __init__(self, inverted: bool = False, *args, **kwargs):
        """
        inverted=True for y-axis pass
        """
        super().__init__(*args, **kwargs)
        self._inverted = inverted

    def add(self, major: float, minor: float, length: float = 1):
        """
        add a wall segment in row <major> at position <minor> with length <length> and merge with previous line segment if their endpoints touch
        """
        if major not in self:
            self[major] = [(minor, minor+length)]
            return

        last = self[major][-1]

        if minor == last[1]:
            self[major][-1] = (last[0], minor+length)
        else:
            self[major].append((minor, minor+length))

    @property
    def lines(self) -> WorldWalls:
        """
        get WorldWalls object
        """
        if not self._inverted:
            return set([((int(start), int(major)), (int(end), int(major))) for major, segment in self.items() for start, end in segment])
        else:
            return set([((int(major), int(start)), (int(major), int(end))) for major, segment in self.items() for start, end in segment])

def RLE_1D(grid: np.ndarray) -> List[List[int]]:
    """
    run-length encode walls in 1D (occupancy grid -> run_length[segments][rows])
    """
    res: List[List[int]] = list()
    for major in grid:
        run: int = 1
        last: int = major[0]
        subres: List[int] = [0]
        for minor in major[1:]:
            if minor == last:
                run += 1
            else:
                subres.append(run)
                run = 1
                last = minor
        subres.append(run)
        res.append(subres)
    return res

def RLE_2D(grid: np.ndarray) -> WorldWalls:
    """
    rudimentary (but fast) 2D extension of 1D-RLE to 2D (occupancy grid -> WorldWalls)
    """

    walls_x = _WallLines()
    walls_y = _WallLines(inverted=True)

    for y, rles in enumerate(RLE_1D(grid)):
        distance: int = 0
        for run in rles:
            distance += run
            walls_x.add(distance, y)

    for x, rles in enumerate(RLE_1D(grid.T)):
        distance: int = 0
        for run in rles:
            distance += run
            walls_y.add(distance, x)

    return set().union(walls_x.lines, walls_y.lines)

@dataclasses.dataclass
class WorldObstacleConfiguration:
    """
    only use this for receiving ros messages
    """
    position: PositionOrientation
    model_name: str
    extra: Dict


@dataclasses.dataclass
class WorldEntities:
    obstacles: WorldObstacles
    walls: WorldWalls


class WorldOccupancy:
    FULL: np.uint8 = np.uint8(np.iinfo(np.uint8).min)
    EMPTY: np.uint8 = np.uint8(np.iinfo(np.uint8).max)

    _grid: np.ndarray

    def __init__(self, grid: np.ndarray):
        self._grid = grid

    @staticmethod
    def from_map(input_map: np.ndarray) -> "WorldOccupancy":
        remap = scipy.interpolate.interp1d([input_map.max(),input_map.min()],[WorldOccupancy.EMPTY, WorldOccupancy.FULL])
        return WorldOccupancy(remap(input_map))

    @staticmethod
    def empty(grid: np.ndarray) -> np.ndarray:
        return grid >= (WorldOccupancy.FULL + WorldOccupancy.EMPTY) / 2
    
    @staticmethod
    def not_empty(grid: np.ndarray) -> np.ndarray:
        return grid < WorldOccupancy.EMPTY
    
    @staticmethod
    def full(grid: np.ndarray) -> np.ndarray:
        return grid <= (WorldOccupancy.FULL + WorldOccupancy.EMPTY) / 2
    
    @staticmethod
    def not_full(grid: np.ndarray) -> np.ndarray:
        return grid > WorldOccupancy.FULL

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    def clear(self):
        self.grid.fill(WorldOccupancy.EMPTY)

    def occupy(self, zone: Position, radius: float):
        self._grid[
            int(zone[1]-radius):int(zone[1]+radius),
            int(zone[0]-radius):int(zone[0]+radius)
        ] = WorldOccupancy.FULL
