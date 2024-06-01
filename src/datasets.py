import numpy as np
from andi_datasets.models_phenom import models_phenom
from andi_datasets.utils_challenge import label_continuous_to_list


class AndiDataset:

    def __init__(self):
        self.L = 1.5*128
        self.model_phenon = models_phenom()
        self.number_compartments = 50
        self.radius_compartments = 10
        self.D_range = np.logspace(-6, 3, 100_000)
    
    @staticmethod
    def random_alpha_value():
        return np.random.uniform(0, 2)

    @staticmethod
    def generate_proper_alpha():
        alpha_1, alpha_2 = AndiDataset.random_alpha_value(), AndiDataset.random_alpha_value()
        while abs(alpha_1 - alpha_2) < 0.05:
            alpha_1, alpha_2 = AndiDataset.random_alpha_value(), AndiDataset.random_alpha_value()
        return alpha_1, alpha_2
    
    def random_D_value(self):
        return (np.random.choice(self.D_range) * np.random.choice([np.random.uniform(0, 1), np.random.randint(1)], size=1))[0]
    
    def get_alphas_and_Ds(self):
        alpha_1, alpha_2 = self.generate_proper_alpha()
        D_1, D_2 = self.random_D_value(), self.random_D_value()
        return [[alpha_1, np.random.uniform(0, 0.15)], [alpha_2, np.random.uniform(0, 0.15)]], \
               [[D_1, np.random.uniform(0, 0.15)],  [D_2, np.random.uniform(0, 0.15)]]
    
    def single_state(self, N, T, alphas, Ds):
        return self.model_phenon.single_state(N=N, T=T, alphas=alphas, Ds=[Ds, np.random.uniform(0.01, 0.4)])

    def multi_state(self, N, T, alphas, Ds):
        p = np.random.uniform(0.02, 0.15)
        p2 = np.random.uniform(0.02, 0.15)
        return self.model_phenon.multi_state(N=N, T=T, M=[[1-p, p], [p2, 1-p2]], Ds=Ds, alphas=alphas)

    def confinemnet(self, N, T, alphas, Ds):
        compartments_center = models_phenom._distribute_circular_compartments(Nc=self.number_compartments, 
                                                                              r=self.radius_compartments, 
                                                                              L=self.L
                                                                              ) 
        return self.model_phenon.confinement(N=N, 
                                             T=T,
                                             Ds=Ds, 
                                             alphas=alphas, 
                                             comp_center=compartments_center,
                                             trans=0.2,
                                             r=self.radius_compartments
                                             )

    def immobile(self, N, T, alphas, Ds):
        Pb = np.random.uniform(0.85, 1)
        Pu = np.random.uniform(0, 0.15)
        number_traps = np.random.randint(50, 150)
        traps_positions = np.random.rand(number_traps, 2) * self.L 
        return self.model_phenon.immobile_traps(N=N,
                                                T=T,
                                                L=self.L,
                                                r=2,
                                                Pb=Pb,
                                                Pu=Pu,
                                                Ds=Ds,
                                                alphas=alphas,
                                                Nt=number_traps,
                                                traps_pos=traps_positions
                                                )

    def dimmerization(self, N, T, alphas, Ds):
        Pb = np.random.uniform(0.9, 1)
        Pu = np.random.uniform(0, 0.1)
        return self.model_phenon.dimerization(N=N,
                                              L=self.L,
                                              T=T,
                                              alphas=alphas,
                                              Ds=Ds,
                                              r=1,
                                              Pb=Pb,
                                              Pu=Pu
                                              )
    
    def get_changepoints(self, labels):
        changepoints_list = []
        for i in range(labels.shape[1]):
            changepoints = label_continuous_to_list(labels[:, i, :])[0]
            changepoints_list.append(changepoints)
        return changepoints_list
