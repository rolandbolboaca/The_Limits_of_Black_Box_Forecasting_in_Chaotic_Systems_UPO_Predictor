import pickle
import pandas as pd
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class DatasetConfig:
    sim_step: int
    simulations: int
    simulation_len: int
    data: str
    data_test: str

class DataHandler:


    DATASETS = { 

        # ---------- 0.05 ----------
        "dt005_sim1_all_rhos_1000": DatasetConfig(
            simulation_len = 99_901,
            sim_step=99_901,
            simulations=1,
            data="data/Norm_IC_01_G_Rho_5_1_225_005_Sim_1_27_01_2026_13_03_38.csv",
            data_test="data/Norm_IC_01_G_Rho_5_1_225_005_Sim_1_27_01_2026_13_03_52.csv",
        ),


        "dt005_rho_28_10": DatasetConfig(
            simulation_len = 2000,
            sim_step=20_000,
            simulations=10,
            data="data/Norm_IC_-1515_G_Rho_28_1_28___Sim_10_30_01_2026_10_06_96.csv",
            data_test="data/Norm_IC_-1515_G_Rho_28_1_28___Sim_10_30_01_2026_10_06_96.csv",
        ),


        "dt005_rho_28_10_10000": DatasetConfig(
            simulation_len = 200_000,
            sim_step=2_000_000,
            simulations=10,
            data="data/Norm_IC_01_G_Rho_28_28_28_005_Sim_10_02_02_2026_11_32_26.csv",
            data_test="data/Norm_IC_01_G_Rho_28_28_28_005_Sim_10_02_02_2026_11_32_38.csv",
        ),
        
        "dt005_rho_28_10_50000_2": DatasetConfig(
            simulation_len = 1_000_000,
            sim_step=10_000_000,
            simulations=10,
            data="data/Norm_IC_01_G_Rho_28_28_28_005_Sim_10_02_02_2026_12_07_21.csv",
            data_test="data/Norm_IC_01_G_Rho_28_28_28_005_Sim_10_02_02_2026_12_07_84.csv",
        ),

        "dt005_sim1_all_rhos_5000": DatasetConfig(
            simulation_len = 499_901,
            sim_step=499_901,
            simulations=1,
            data="data/Norm_IC_01_G_Rho_5_1_225_005_Sim_1_27_01_2026_13_27_90.csv",
            data_test="data/Norm_IC_01_G_Rho_5_1_225_005_Sim_1_27_01_2026_13_27_94.csv",
        ),

        "dt005_sim10_all_rhos_500": DatasetConfig(
            simulation_len = 10_000,
            sim_step=100_000,
            simulations=10,
            data="data/Norm_IC_01_G_Rho_5_1_225_005_Sim_10_29_01_2026_11_32_41.csv",
            data_test="data/Norm_IC_01_G_Rho_5_1_225_005_Sim_10_29_01_2026_11_32_44.csv",
        ),

        "dt005_sim10_all_rhos_1000": DatasetConfig(
            simulation_len = 20_000,
            sim_step=200_000,
            simulations=10,
            data="data/Norm_IC_01_G_Rho_5_1_225_005_Sim_10_29_01_2026_16_11_52.csv",
            data_test="data/Norm_IC_01_G_Rho_5_1_225_005_Sim_10_29_01_2026_16_11_95.csv",
        ),

        "dt005_sim10_all_rhos_1000_2": DatasetConfig(
            simulation_len = 20_000,
            sim_step=200_000,
            simulations=10,
            data="data/Norm_IC_01_G_Rho_5_1_225_005_Sim_10_30_01_2026_09_33_40.csv",
            data_test="data/Norm_IC_01_G_Rho_5_1_225_005_Sim_10_30_01_2026_09_34_55.csv",
        ),

        "dt005_sim10": DatasetConfig(
            simulation_len = 1981,
            sim_step=19_810,
            simulations=10,
            data="data/Norm_IC_01_G_Rho_5_5_225___Sim_10_16_10_2025_07_34_19.csv",
            data_test="data/Norm_IC_01_G_Rho_5_5_225___Sim_10_16_10_2025_07_34_93.csv",
        ),

        "dt005_sim10_rhoincr1": DatasetConfig(
            simulation_len = 1981,
            sim_step=19_810,
            simulations=10,
            data="data/Norm_IC_01_G_Rho_5_1_225___Sim_10_08_01_2026_15_46_46.csv",
            data_test="data/Norm_IC_01_G_Rho_5_1_225___Sim_10_08_01_2026_15_46_96.csv",
        ),

        "dt005_sim20": DatasetConfig(
            simulation_len = 1981,
            sim_step=39_620,
            simulations=20,
            data="data/Norm_IC_01_G_Rho_5_5_225_005_Sim_20_31_12_2025_10_47_07.csv",
            data_test="data/Norm_IC_01_G_Rho_5_5_225_005_Sim_20_31_12_2025_10_47_45.csv",
        ),

        "dt005_sim50": DatasetConfig(
            simulation_len = 1981,
            sim_step=99_050,
            simulations=50,
            data="data/Norm_IC_01_G_Rho_5_5_225_005_Sim_50_31_12_2025_10_52_25.csv",
            data_test="data/Norm_IC_01_G_Rho_5_5_225_005_Sim_50_31_12_2025_10_52_50.csv",
        ),

        "dt005_sim100": DatasetConfig(
            simulation_len = 1981,
            sim_step=198_100,
            simulations=100,
            data="data/Norm_IC_01_G_Rho_5_5_225_005_Sim_100_30_12_2025_16_54_14.csv",
            data_test="data/Norm_IC_01_G_Rho_5_5_225_005_Sim_100_30_12_2025_16_54_32.csv",
        ),
        "dt005_all_rho_sim100": DatasetConfig(
            simulation_len = 2000,
            sim_step=200_000,
            simulations=100,
            data="data/Norm_IC_01_G_Rho_5_1_225_005_Sim_100_19_02_2026_11_51_56.csv",
            data_test="data/Norm_IC_01_G_Rho_5_1_225_005_Sim_100_19_02_2026_11_52_51.csv",
        ),

        # ---------- 0.01 ----------

        "dt001_sim1_all_rhos_10000": DatasetConfig(
            simulation_len = 999_901,
            sim_step=999_901,
            simulations=1,
            data="data/Norm_IC_01_G_Rho_5_1_225_001_Sim_1_27_01_2026_15_18_48.csv",
            data_test="data/Norm_IC_01_G_Rho_5_1_225_001_Sim_1_27_01_2026_15_18_70.csv",
        ),

        "dt001_sim1_all_rhos_5000": DatasetConfig(
            simulation_len = 499_901,
            sim_step=499_901,
            simulations=1,
            data="data/Norm_IC_01_G_Rho_5_1_225_001_Sim_1_27_01_2026_13_54_34.csv",
            data_test="data/Norm_IC_01_G_Rho_5_1_225_001_Sim_1_27_01_2026_13_54_98.csv",
        ),

        "dt001_sim1_all_rhos_1000": DatasetConfig(
            simulation_len = 99_901,
            sim_step=99_901,
            simulations=1,
            data="data/Norm_IC_01_G_Rho_5_1_225_001_Sim_1_27_01_2026_12_56_26.csv",
            data_test="data/Norm_IC_01_G_Rho_5_1_225_001_Sim_1_27_01_2026_12_56_62.csv",
        ),


        "dt001_sim10_all_rhos_500": DatasetConfig(
            simulation_len = 50_000,
            sim_step=500_000,
            simulations=10,
            data="data/Norm_IC_01_G_Rho_5_1_225_001_Sim_10_12_02_2026_12_31_65.csv",
            data_test="data/Norm_IC_01_G_Rho_5_1_225_001_Sim_10_12_02_2026_12_31_78.csv",
        ),

        "dt001_sim10_all_rhos_1000": DatasetConfig(
            simulation_len = 100_000,
            sim_step=1_000_000,
            simulations=10,
            data="data/ .csv",
            data_test="data/ .csv",
        ),

        "dt001_sim10": DatasetConfig(
            simulation_len = 9901,
            sim_step=99_010,
            simulations=10,
            data="data/Norm_IC_01_G_Rho_5_5_225_001_Sim_10_31_12_2025_11_09_49.csv",
            data_test="data/Norm_IC_01_G_Rho_5_5_225_001_Sim_10_31_12_2025_11_09_85.csv",
        ),

        "dt001_sim10_all_rhos": DatasetConfig(
            simulation_len = 9901,
            sim_step=99_010,
            simulations=10,
            data="data/Norm_IC_01_G_Rho_5_1_225_001_Sim_10_23_01_2026_12_03_47.csv",
            data_test="data/Norm_IC_01_G_Rho_5_1_225_001_Sim_10_23_01_2026_12_03_83.csv",
        ),

        "dt001_sim20": DatasetConfig(
            simulation_len = 9901,
            sim_step=198_020,
            simulations=20,
            data="data/Norm_IC_01_G_Rho_5_5_225_001_Sim_20_30_12_2025_12_44_42.csv",
            data_test="data/Norm_IC_01_G_Rho_5_5_225_001_Sim_20_30_12_2025_12_44_87.csv",
        ),

        "dt001_sim100": DatasetConfig(
            simulation_len = 9901,
            sim_step=990_100,
            simulations=100,
            data="data/Norm_IC_01_G_Rho_5_5_225_001_Sim_100_30_12_2025_14_06_07.csv",
            data_test="data/Norm_IC_01_G_Rho_5_5_225_001_Sim_100_30_12_2025_14_06_40.csv",
        ),


        # ---------- 0.03 ----------
        "dt003_sim10": DatasetConfig(
            simulation_len = 3301,
            sim_step=33_010,
            simulations=10,
            data="data/Norm_IC_01_G_Rho_5_5_225_003_Sim_10_31_12_2025_11_13_12.csv",
            data_test="data/Norm_IC_01_G_Rho_5_5_225_003_Sim_10_31_12_2025_11_13_70.csv",
        ),
        "dt003_sim20": DatasetConfig(
            simulation_len = 3301,
            sim_step=66_020,
            simulations=20,
            data="data/Norm_IC_01_G_Rho_5_5_225_003_Sim_20_30_12_2025_15_52_02.csv",
            data_test="data/Norm_IC_01_G_Rho_5_5_225_003_Sim_20_30_12_2025_15_52_74.csv",
        ),

        "dt003_sim100": DatasetConfig(
            simulation_len = 3301,
            sim_step=330_100,
            simulations=100,
            data="data/Norm_IC_01_G_Rho_5_5_225_003_Sim_100_30_12_2025_16_14_26.csv",
            data_test="data/Norm_IC_01_G_Rho_5_5_225_003_Sim_100_30_12_2025_16_14_60.csv",
        ),

        "dt003_all_rho_sim100": DatasetConfig(
            simulation_len = 3333,
            sim_step=333_300,
            simulations=100,
            data="data/Norm_IC_01_G_Rho_5_1_225_003_Sim_100_18_02_2026_20_35_15.csv",
            data_test="data/Norm_IC_01_G_Rho_5_1_225_003_Sim_100_18_02_2026_20_35_79.csv",
        ),

        # ---------- mixed / special ----------
        "mixed_sim10": DatasetConfig(
            simulation_len = 2223,
            sim_step=2223,
            simulations=10,
            data="data/Norm_IC_-55_G_Rho_5_5_225___Sim_10_15_10_2025_14_03_05.csv",
            data_test="data/Norm_IC_-55_G_Rho_5_5_225___Sim_10_15_10_2025_14_03_99.csv",
        ),

        "mixed_sim100": DatasetConfig(
            simulation_len = 2223,
            sim_step=2223,
            simulations=100,
            data="data/Norm_IC_-55_G_Rho_5_5_225___Sim_100_15_10_2025_12_54_07.csv",
            data_test="data/Norm_IC_-55_G_Rho_5_5_225___Sim_100_15_10_2025_12_54_52.csv",
        ),
        "mixed_sim100_s40": DatasetConfig(
            simulation_len = 9981,
            sim_step=9981,
            simulations=100,
            data="data/Norm_IC_01_G_Rho_25_40_225__RS_Sim_100_06_01_2026_13_41_43.csv",
            data_test="data/Norm_IC_01_G_Rho_25_40_225__RS_Sim_100_06_01_2026_13_41_86.csv",
        ),
        "mixed_sim10_s40_Increasing_Rho": DatasetConfig(
            simulation_len = 9981,
            sim_step=9981,
            simulations=10,
            data="data/Norm_IC_01_G_Rho_25_40_225__RS_Sim_10_06_01_2026_13_10_81.csv",
            data_test="data/Norm_IC_01_G_Rho_25_40_250__RS_Sim_10_06_01_2026_13_03_60.csv",
        ),
        "mixed_sim10_s40": DatasetConfig(
            simulation_len = 9981,
            sim_step=9981,
            simulations=10,
            data="data/Norm_IC_01_G_Rho_25_40_225__RS_Sim_10_06_01_2026_13_56_88.csv",
            data_test="data/Norm_IC_01_G_Rho_25_40_225__RS_Sim_10_06_01_2026_13_56_90.csv",
        ),
        "mixed_sim10_s40_nonoise": DatasetConfig(
            simulation_len = 9981,
            sim_step=9981,
            simulations=10,
            data="data/Norm_IC_01_ZERO_Rho_25_40_225__RS_Sim_10_06_01_2026_13_24_24.csv",
            data_test="data/Norm_IC_01_ZERO_Rho_25_40_225__RS_Sim_10_06_01_2026_13_24_31.csv",
        ),

    }
    @staticmethod
    def save_pickle(data, data_path):
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load_pickle(data_path):
        with open(data_path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def load_csv(data_path, sep=',', decimal='.', quotechar='"', encoding=None):
        import numpy as np
        dtypes = {
            0: np.int32,   # Simulation
            1: np.int16,   # Sub-sim
            2: np.float32, # Time
            3: np.float32, # X
            4: np.float32, # Y
            5: np.float32, # Z
            6: np.float32  # RHO
        }
        return pd.read_csv(data_path, 
                           sep=sep,
                           decimal=decimal,
                           quotechar=quotechar,
                           encoding=encoding,
                            #header = None,
                            dtype=dtypes,
                            engine='c',
                            na_filter=False,      
                            low_memory=False      
                           )
    @staticmethod
    def save_csv(data, data_path, sep=',', decimal='.', quotechar='"', encoding=None, index=False):
        pd.DataFrame(data).to_csv(data_path, 
                                  sep=sep,
                                  decimal=decimal,
                                  quotechar=quotechar,
                                  encoding=encoding,
                                  index=index)

    @staticmethod
    def load_experiment(name: str):
        """
        Returns sim_step, SIMULATIONS, data, data_test
        """
        if name not in DataHandler.DATASETS:
            raise ValueError(
                f"Unknown experiment '{name}'. Available:\n"
                + "\n".join(DataHandler.DATASETS.keys())
            )

        cfg = DataHandler.DATASETS[name]
        return cfg.simulation_len, cfg.sim_step, cfg.simulations, cfg.data, cfg.data_test

    @staticmethod
    def limit_simulations(data, max_sims):
        if max_sims == 0:
            return data  

        result_list = []
        unique_rhos = np.unique(data[:, 6])

        for rho in unique_rhos:
            rho_rows = data[data[:, 6] == rho]
            sim_ids = np.unique(rho_rows[:, 0])
            selected_sims = sim_ids[:max_sims]
            mask = np.isin(rho_rows[:, 0], selected_sims)
            result_list.append(rho_rows[mask])

        return np.vstack(result_list)

    @staticmethod
    def remove_transient(data, sim_col=0, rho_col=6, II=100):
        out = []

        for rho in np.unique(data[:, rho_col]):
            rho_data = data[data[:, rho_col] == rho]

            for sim in np.unique(rho_data[:, sim_col]):
                sim_data = rho_data[rho_data[:, sim_col] == sim]

                if sim_data.shape[0] > II:
                    out.append(sim_data[II:]) 

        return np.vstack(out)

    @staticmethod
    def load_saved_parquets(data_path, columns=None):
        return pd.read_parquet(data_path, columns=columns)
    
    @staticmethod
    def load_exp_data_(EXPERIMENT_NAME="dt003_all_rho_sim100",
                    MAX_SIMULATIONS_TRAIN=100,
                    MAX_SIMULATIONS_TEST=100,
                    TRANSIENT_LEN=500,
                    datasets_parameters=None,
                    NAME="LORENZ",
                    DT=0.03):

        import os
        import pandas as pd
        import numpy as np

        if datasets_parameters is None:
            raise ValueError(
                f"Unknown experiment '{EXPERIMENT_NAME}'. Available:\n"
                + "\n".join(DataHandler.DATASETS.keys())
            )

        if EXPERIMENT_NAME.startswith("dt"):
            dt_token = EXPERIMENT_NAME.split("dt")[1].split("_")[0]
            DT_string = format(DT, 'g').replace('.', '')
            assert dt_token == DT_string, (
                f"DT mismatch! File DT token='{dt_token}' "
                f"but current DT corresponds to '{DT_string}'"
            )

        SIMULATION_LEN, sim_step, SIMULATIONS, data, data_test = \
            DataHandler.load_experiment(EXPERIMENT_NAME)

        os.makedirs("data", exist_ok=True)

        cache_train = f"data/{EXPERIMENT_NAME}_train_" \
                    f"tr{MAX_SIMULATIONS_TRAIN}_te{MAX_SIMULATIONS_TEST}_" \
                    f"trans{TRANSIENT_LEN}.parquet"

        cache_test = f"data/{EXPERIMENT_NAME}_test_" \
                    f"tr{MAX_SIMULATIONS_TRAIN}_te{MAX_SIMULATIONS_TEST}_" \
                    f"trans{TRANSIENT_LEN}.parquet"

        if os.path.exists(cache_train) and os.path.exists(cache_test):

            train = pd.read_parquet(cache_train).to_numpy()
            test = pd.read_parquet(cache_test).to_numpy()

            SIMULATION_LEN = SIMULATION_LEN - TRANSIENT_LEN
            SIMULATION_LEN += 1  # Include the last point after transient
            all_data = np.vstack((train, test))

            return (SIMULATION_LEN, sim_step, SIMULATIONS,
                    train, test, all_data)

        train = DataHandler.load_csv(
            data,
            sep=datasets_parameters[NAME]['sep'],
            decimal=datasets_parameters[NAME]['decimal'],
            quotechar='"',
            encoding=datasets_parameters[NAME].get('encoding', None)
        ).to_numpy()

        test = DataHandler.load_csv(
            data_test,
            sep=datasets_parameters[NAME]['sep'],
            decimal=datasets_parameters[NAME]['decimal'],
            quotechar='"',
            encoding=datasets_parameters[NAME].get('encoding', None)
        ).to_numpy()

        if MAX_SIMULATIONS_TRAIN > 0:
            train = DataHandler.limit_simulations(train, MAX_SIMULATIONS_TRAIN)

        if MAX_SIMULATIONS_TEST > 0:
            test = DataHandler.limit_simulations(test, MAX_SIMULATIONS_TEST)

        train = DataHandler.remove_transient(train, II=TRANSIENT_LEN)
        test = DataHandler.remove_transient(test, II=TRANSIENT_LEN)

        SIMULATION_LEN = SIMULATION_LEN - TRANSIENT_LEN
        SIMULATION_LEN += 1  # Include the last point after transient
        all_data = np.vstack((train, test))

        pd.DataFrame(train).to_parquet(cache_train, index=False)
        pd.DataFrame(test).to_parquet(cache_test, index=False)

        return (SIMULATION_LEN, sim_step, SIMULATIONS,
                train, test, all_data)