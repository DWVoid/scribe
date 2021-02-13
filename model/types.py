from typing import *
import numpy as np


# stores the meta-data of the compiled Set. if this changes, the Set needs to be re-compiled
class DataSetMeta:
    limit: int
    t_steps: int
    alphabet: str
    data_scale: int
    ascii_steps: int
    sample_space_size: int

    def transfer(self, args, limit: int) -> None:
        self.limit = limit  # removes large noisy gaps in the data
        self.t_steps = args.tsteps
        self.alphabet = args.alphabet
        self.data_scale = args.data_scale  # scale data down by this factor
        self.ascii_steps = int(args.tsteps / args.tsteps_per_ascii)
        self.sample_space_size = args.batch_size * args.nbatches

    def __eq__(self, other) -> bool:
        return self.limit == other.limit \
               and self.t_steps == other.t_steps \
               and self.alphabet == other.alphabet \
               and self.data_scale == other.data_scale \
               and self.ascii_steps == other.ascii_steps \
               and self.sample_space_size == other.sample_space_size


DataSetRaw = List[Tuple[np.ndarray, str]]
DataSetsRaw = Dict[int, DataSetRaw]

DataSetShaped = Tuple[Dict[str, List[np.ndarray]], List[np.ndarray]]
DataSetCompiled = Tuple[DataSetShaped, DataSetShaped]
DataSetsCompiled = Dict[int, DataSetCompiled]
