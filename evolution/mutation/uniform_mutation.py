from evolution.candidate import Candidate
from evolution.mutation.mutation import Mutation

class UniformMutation(Mutation):
    """
    Uniformly mutates 
    """
    def __init__(self, mutation_factor, mutation_rate):
        super().__init__(mutation_factor, mutation_rate)
        self.type = "uniform"

    def mutate_(self, candidate: Candidate):
        """
        Mutate model with gaussian percentage in-place
        """
        for param in candidate.model.parameters():
            self.gaussian_pct_(param.data)