import argparse
import fair
import gym
import json
import math
import os
import random
import sys
import time
import torch
import pandas as pd 

import matplotlib.pyplot as plt
import numpy as np

#from fair.RCPs import rcp26, rcp45, rcp60, rcp85
from fair.SSPs import ssp370, ssp245, ssp585
from functools import partial
from scipy.stats import gamma
from stable_baselines3 import PPO, DDPG, A2C

#SSP 370 CO2 Emissions from 2021 to 2098+300
ssp_370 = [12.447165727955467, 12.66655279064781, 12.885939853340151, 13.105326916032494, 13.324713978724835, 13.54410104141718, 13.763488104109522, 13.982875166801863, 14.202262229494206, 14.42164929218655, 14.575850228958055, 14.730051165729563, 14.884252102501067, 15.038453039272575, 15.192653976044081, 15.346854912815589, 15.501055849587095, 15.655256786358603, 15.809457723130109, 15.963658659901615, 16.08389752920384, 16.20413639850607, 16.324375267808296, 16.444614137110523, 16.564853006412747, 16.685091875714974, 16.805330745017205, 16.92556961431943, 17.045808483621652, 17.166047352923883, 17.26604360611622, 17.366039859308565, 17.46603611250091, 17.56603236569325, 17.666028618885594, 17.76602487207793, 17.866021125270276, 17.96601737846262, 18.06601363165496, 18.166009884847305, 18.260802131031195, 18.355594377215084, 18.45038662339898, 18.545178869582866, 18.639971115766755, 18.734763361950648, 18.829555608134537, 18.924347854318423, 19.01914010050232, 19.113932346686205, 19.205712852756513, 19.297493358826813, 19.389273864897113, 19.481054370967417, 19.57283487703772, 19.66461538310802, 19.75639588917832, 19.848176395248625, 19.939956901318933, 20.03173740738923, 20.15164156606908, 20.27154572474892, 20.39144988342877, 20.51135404210861, 20.63125820078846, 20.751162359468307, 20.871066518148155, 20.990970676827995, 21.110874835507836, 21.230778994187688, 21.365227238307206, 21.49967548242672, 21.63412372654624, 21.768571970665757, 21.90302021478528, 22.037468458904797, 22.171916703024312, 22.306364947143834, 22.440813191263352, 22.57526143538287, 22.415095217954555, 22.254929000526243, 22.094762783097938, 21.93459656566962, 21.77443034824131, 21.614264130812998, 21.45409791338469, 21.293931695956374, 21.13376547852806, 20.97359926109975, 20.81343304367144, 20.653266826243126, 20.493100608814814, 20.332934391386505, 20.17276817395819, 20.01260195652988, 19.852435739101566, 19.692269521673257, 19.532103304244945, 19.371937086816637, 19.211770869388324, 19.051604651960005, 18.8914384345317, 18.731272217103385, 18.571105999675073, 18.41093978224676, 18.250773564818452, 18.09060734739014, 17.930441129961824, 17.770274912533512, 17.610108695105204, 17.449942477676892, 17.28977626024858, 17.12961004282027, 16.969443825391956, 16.809277607963644, 16.64911139053533, 16.488945173107023, 16.32877895567871, 16.1686127382504, 16.008446520549192, 15.848280302847993, 15.688114085146786, 15.527947867445581, 15.367781649744375, 15.207615432043173, 15.047449214341968, 14.887282996640765, 14.727116778939559, 14.566950561238354, 14.421281055653262, 14.275611550068167, 14.129942044483075, 13.984272538897978, 13.838603033312882, 13.69293352772779, 13.547264022142697, 13.401594516557601, 13.25592501097251, 13.110255505387412, 12.964585999802317, 12.818916494217225, 12.673246988632128, 12.527577483047036, 12.38190797746194, 12.236238471876847, 12.090568966291752, 11.944899460706656, 11.799229955121563, 11.653560449536467, 11.507890943951374, 11.36222143836628, 11.216551932781185, 11.070882427196091, 10.925212921611, 10.779543416025902, 10.633873910440808, 10.488204404855715, 10.342534899270621, 10.196865393685526, 10.051195888100432, 9.905526382515339, 9.759856876930243, 9.61418737134515, 9.468517865760056, 9.322848360174959, 9.177178854589865, 9.031509349004772, 8.885839843419678, 8.740170337834583, 8.594500832249489, 8.448831326664395, 8.3031618210793, 8.157492315494206, 8.011822809909113, 7.866153304324018, 7.720483798738924, 7.574814293153827, 7.429144787568734, 7.283475281983639, 7.137805776398546, 6.992136270813451, 6.846466765228357, 6.700797259643263, 6.5551277540581685, 6.409458248473075, 6.2637887428879795, 6.118119237302886, 5.972449731717792, 5.826780226132698, 5.68111072027471, 5.535441214416724, 5.3897717085587376, 5.24410220270075, 5.098432696842763, 4.9527631909847765, 4.80709368512679, 4.661424179268803, 4.515754673410816, 4.370085167552829, 4.224415661967735, 4.078746156382641, 3.9330766507975463, 3.787407145212452, 3.641737639627358, 3.4960681340422632, 3.350398628457169, 3.2047291228720742, 3.059059617286981, 2.913390111701886, 2.7677206061440818, 2.6220511005862757, 2.4763815950284704, 2.330712089470666, 2.1850425839128613, 2.039373078355056, 1.893703572797251, 1.7480340672394457, 1.6023645616816409, 1.4566950561238354, 1.3110255505114519, 1.1653560448990683, 1.0196865392866847, 0.8740170336743013, 0.7283475280619177, 0.582678022449534, 0.4370085168371506, 0.291339011224767, 0.14566950561238337, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]




#Economic dimension
Y= np.zeros (356+300) #global GDP
Y_cost = np.zeros (356+300) #cost of climate change
S= np.zeros (356+300) #renewable knowledge stock
S[255] = 5e11 #GJ
Y[255] = 9e13 #USD/a
Y_cost[255] = 100*1e9 #USD/a
S[256] = 5e11 #GJ
Y[256] = 9e13 #USD/a
Y_cost[256] = 100*1e9 #USD/a
beta = 0.03 # 1/yr
epsilon = 147.  # USD/GJ
rho = 2.  # 1
sigma = 4.e12  # GJ
tau_S = 65.  # yr

# Labels for plots
VARS = [
    "Temperature anomaly (ºC)", 
    "CO2 Emissions (GtC)", 
    "CO2 Concentration (ppm)", 
    "Radiative forcing (W m^-2)",
    "Reward "
]
MULTIGAS_VARS = [
    "Temperature anomaly (ºC)", 
    "CO2 Emissions (GtC)", 
    "CO2 Concentration (ppm)", 
    "CO2 forcing (W m^-2)",
    "CH4 forcing (W m^-2)",
    "N2O forcing (W m^-2)",
    "All other well-mixed GHGs forcing (W m^-2)",
    "Tropospheric O3 forcing (W m^-2)",
    "Stratospheric O3 forcing (W m^-2)",
    "Stratospheric water vapour from CH4 oxidation forcing (W m^-2)",
    "Contrails forcing (W m^-2)",
    "Aerosols forcing (W m^-2)",
    "Black carbon on snow forcing (W m^-2)",
    "Land use change forcing (W m^-2)",
    "Volcanic forcing (W m^-2)",
    "Solar forcing (W m^-2)",
    "Reward "
]


#### Reward function options ####

def simple_reward(state, cur_temp, t, cur_emit, cur_conc, cur_GDP, cur_fease):
    # positive reward for temp decrease
    # negative cliff if warming exceeds 2º
    if cur_temp > 2:
        return -100
    return (state[0] - cur_temp)

def temp_reward(state, cur_temp, t, cur_emit, cur_conc, cur_GDP, cur_fease):
    # positive reward for temp under 1.5 goal
    if cur_temp > 2:
        return -100
    return 1.5 - cur_temp

def conc_reward(state, cur_temp, t, cur_emit, cur_conc, cur_GDP, cur_fease):
    # positive reward for decreased concentration
    if cur_temp > 2:
        return -100
    return state[2] - cur_conc

def carbon_cost_reward(state, cur_temp, t, cur_emit, cur_conc, cur_GDP, cur_fease):
    # impose a cost for each GtC emitted
    if cur_temp > 2:
        return -100
    return -cur_emit


def carbon_cost_GDP_reward(state, cur_temp, t, cur_emit, cur_conc, cur_GDP, cur_fease):
    sum = 0
    if cur_temp > 2:
        sum += -1000
    if cur_fease < 0 : 
        sum += -1000
    if cur_fease > 0:
        sum += -10*cur_fease
    sum += -cur_PIB/1e11 
    return sum 


def temp_emit_reward(state, cur_temp, t, cur_emit, cur_conc, cur_GDP, cur_fease):
    # positive reward for keeping the temp under 1.5
    # negative reward for amount of emissions reduction
    # positive cliff for success at the end of the trial
    # w could indicate cost
    if cur_temp > 2:
        return -100
    if t==79 and temp <=1.5:
        return 100
    temp = 10*(state[0] - cur_temp)
    emit = state[1] - cur_emit
    if cur_emit < state[1]:
        return temp - emit
    return temp

def temp_emit_diff_reward(state, cur_temp, t, cur_emit, cur_conc, cur_GDP, cur_fease):
    # positive reward for keeping the temp under 1.5
    # negative reward for amount of emissions reduction
    # (reduction compared to projected amount for that year)
    # positive cliff for success at the end of the trial
    # w could indicate cost of emissions
    if cur_temp > 2:
        return -100
    if t==79 and temp <=1.5:
        return 100
    curval = t*0.6 + 36
    temp = 10*(state[0] - cur_temp)
    emit = curval - cur_emit
    if cur_emit < curval:
        return temp - emit
    return temp


#### Environment for FaIR simulator ####
# built to run with the OpenAI Gym

class Simulator(gym.Env):

    def __init__(
        self,
        verbose=1, 
        action_space=36, 
        reward_mode="carbon_cost_PIB", 
        forcing=False,
        multigas=True,
        scenario='ssp245'
    ):
        # action space for the environment,
        # the amount to increase or decrease emissions by
        self.action_space = gym.spaces.Box(
            np.array([-action_space]).astype(np.float32),
            np.array([+action_space]).astype(np.float32),
        )

        # state space, [temperature, carbon emissions, carbon concentration, radiative forcing]
        if not multigas:
            self.observation_space = gym.spaces.Box(
                np.array([-100, -100, 0, -100]).astype(np.float32),
                np.array([100, 100, 5000, 100]).astype(np.float32),
            )
        else:
            self.observation_space = gym.spaces.Box(
                np.array([-100, -100, 0, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50]).astype(np.float32),
                np.array([100, 100, 5000, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]).astype(np.float32),
            )

        # specify the reward function to use
        if reward_mode == "simple":
            self.reward_func = simple_reward
        elif reward_mode == "temp":
            self.reward_func = temp_reward
        elif reward_mode == "conc":
            self.reward_func = conc_reward
        elif reward_mode == "carbon_cost":
            self.reward_func = carbon_cost_reward
        elif reward_mode == "temp_emit":
            self.reward_func = temp_emit_reward
        elif reward_mode == "temp_emit_diff":
            self.reward_func = temp_emit_diff_reward
        elif reward_mode == "carbon_cost_GDP":
            self.reward_func = carbon_cost_GDP_reward

        # setup additional forcing factors
        if forcing:
            solar = 0.1 * np.sin(2 * np.pi * np.arange(736) / 11.5)
            volcanic = -gamma.rvs(0.2, size=736, random_state=14)
            self.forward_func = partial(fair.forward.fair_scm, F_solar=solar, F_volcanic=volcanic)
        else:
            self.forward_func = fair.forward.fair_scm
        self.multigas = multigas
        self.scenario = scenario

        # set the initial state
        self.reset()

    def update_state(self, C, F, T):
         
        if self.multigas:
            concentration = C[:,0][256+self.t]
            forcing = np.sum(F,axis=1)[256+self.t]
            emissions = self.emissions[:,1][256+self.t]
            forcing = [forcing]
        else:
            concentration = C[256+self.t]
            forcing = F[256+self.t] 
            emissions = self.emissions[256+self.t]
            forcing = [forcing]
        self.state = [T[256+self.t], emissions, concentration] + forcing
        self.t += 1
        
    
    def reset(self):
        # initialize historical emissions from SSP scenario
        
        base_emissions = eval(self.scenario).Emissions.emissions 
        if not self.multigas:
            base_emissions = np.array([x[1]+x[2] for x in base_emissions])

        # 80 year time horizon, meet goals by 2100
        self.emissions = base_emissions
        # 2021 estimate of GtC of carbon emissions
        # 2021 is the 257th year in the ssp scenario
           
            
            
        self.t = 0
        # initial state
        C, F, T = self.forward_func(
            emissions=self.emissions, 
            useMultigas=True,
        )
        self.update_state(C, F, T)

        return self.state

    def step(self, action):
        done = False
        
        # change emissions by the action amount
        if not self.multigas: 
            self.emissions[256+self.t] = max(self.emissions[256+self.t-1] + action[0], 0)
        else:
            self.emissions[:,1][256+self.t] = max(self.emissions[:,1][256+self.t-1] + action[0]*.9, 0)
            self.emissions[:,2][256+self.t] = max(self.emissions[:,2][256+self.t-1] + action[0]*.1, 0)

        # run FaIR simulator
        C, F, T = self.forward_func(
            emissions=self.emissions,
            useMultigas= True,
        )
        
        
        #Implementation of S, Y and Y_cost
        gamma = 1 / ( 1+(S[256+self.t-1]/sigma)**rho )
        Y[256+self.t] = Y[256+self.t-1] + beta*Y[256+self.t-1]         
        Y_cost[256+self.t] = (10/5*T[256+self.t]-2)/100*Y[256+self.t]         
        S[256+self.t] = S[256+self.t-1] + ( (1-gamma)*Y[256+self.t-1]/epsilon - S[256+self.t-1]/tau_S )
       
                
        # fail if temperature error
        if math.isnan(T[256+self.t]):
            done = True
        
          
        
        
        if self.multigas:
            cur_emit = self.emissions[:,1][256+self.t] + self.emissions[:,2][256+self.t]
            cur_conc = C[:,0][256+self.t]   
            cur_fease = ssp_370[self.t-1] - cur_emit            
        else :
            # compute the reward
            cur_emit = self.emissions[256+self.t]   
            cur_fease = ssp_370[self.t-1] - self.emissions[256+self.t] 
            cur_conc = C[256+self.t]
        cur_GDP = Y_cost[256+self.t]
                     
            
        reward = self.reward_func(self.state, T[256+self.t], self.t, cur_emit, cur_conc, cur_PIB, cur_fease)
            
        # update the state and info
        self.update_state(C, F, T)

        # end the trial once 2100 is reached
        if self.t == 79 or self.state[0] > 4 or self.state[0] < 0:
            done = True

        return self.state, reward, done, {}

    def render(self, mode="human"):
        # print the state
        print(f'Temperature anomaly: {self.state[0]}ºC')
        print(f'CO2 emissions: {self.state[1]} GtC')
        print(f'CO2 concentration: {self.state[2]} ppm')
        print(f'Radiative forcing: {self.state[3:]}')


#### Training code ####

# Output useful plots
def make_plots(vals, args, save_path):
    plots = MULTIGAS_VARS if args.multigas else VARS
    for i in range(len(plots)):
        name = plots[i][:plots[i].find('(')].strip()
        ys = [x[i] for x in vals]
        xs = [2021+x for x in range(len(vals))]
        plt.plot(xs, ys)
        plt.ylabel(plots[i])
        plt.xlabel('Year')
        plt.savefig(os.path.join(save_path, 'plots', name))
        plt.clf()
   
    
    excel = pd.DataFrame(

        {"Years" : [2021+x for x in range(len(vals))],
         "Temperature anomaly" : [x[0] for x in vals],
         "CO2 Emissions" : [x[1] for x in vals],
         "CO2 Concentration" : [x[2] for x in vals],
         "Radiative forcing" : [x[3] for x in vals],
         "Reward" : [x[4] for x in vals]
        })


    #excel = excel.to_json()
    with open(os.path.join(save_path, 'excel.xls'), 'w') as f:
        excel.to_excel (save_path + 'excel.xls')
        #json.dump(excel, f, indent=None, separators=None) 
    
    


# Main training and eval loop
def main(args, save_path):
    # Create the environment
    env = Simulator(
        action_space = args.action_space,
        reward_mode = args.reward_mode,
        forcing = args.forcing,
        multigas = args.multigas,
        scenario = args.scenario
    )

    # Train the algorithm
    if args.algorithm == 'a2c':
        model_builder = A2C
    elif args.algorithm == 'ppo':
        model_builder = PPO
    elif args.algorithm == 'ddpg':
        model_builder = DDPG
    model = model_builder(
            policy="MlpPolicy",
            env=env, 
            learning_rate = args.lr,
            n_steps = args.n_steps,
            gamma=args.gamma,
            verbose=1,
            device=args.device,
            tensorboard_log=os.path.join(save_path, 'logs'),
    )
    model.learn(
        total_timesteps=args.timesteps,
        eval_freq=20,
        log_interval=20,
        eval_log_path=os.path.join(save_path, 'evals')
    )
    model.save(f'{save_path}/model_state_dict.pt')

    # Run evaluation and make plots
    obs = env.reset()
    vals = []
    for i in range(80):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        vals.append(obs + [reward])
        if done:
          break
    env.close()
    make_plots(vals, args, save_path)


# Make output directories
def make_outdirs(save_path):
    dirs = ['plots', 'logs']
    for dir in dirs:
        path = os.path.join(save_path, dir)
        if not os.path.exists(path):
            os.makedirs(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='test', required=False)
    parser.add_argument("--action_space", type=int, default=2, required=False)
    parser.add_argument("--reward_mode", type=str, default='carbon_cost_GDP', required=False) 
    parser.add_argument("--forcing", action='store_true', required=False)
    parser.add_argument("--output_path", type=str, default='outputs', required=False)
    parser.add_argument("--stdout", action='store_true', required=False)
    parser.add_argument("--seed", type=int, default=random.randint(1,1000), required=False)
    parser.add_argument("--device", type=str, default='cpu', required=False)
    parser.add_argument("--lr", type=float, default=2.1e-5, required=False) 
    parser.add_argument("--n_steps", type=int, default=5, required=False) 
    parser.add_argument("--gamma", type=float, default=0.99, required=False)
    parser.add_argument("--timesteps", type=int, default=10000, required=False) 
    parser.add_argument("--algorithm", type=str, default='a2c', required=False) 
    parser.add_argument("--multigas", action='store_true', required=False) 
    parser.add_argument("--scenario", type=str, default='ssp245', required=False)
    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup save path and logging 
    save_path = os.path.join(args.output_path, args.name)
    make_outdirs(save_path)
    start_time = time.time()
    if not args.stdout:
        sys.stdout = open(os.path.join(save_path, 'stdout.txt'), 'w')
    with open(os.path.join(save_path, 'config.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)


    main(args, save_path)

    # log total runtime and close logging file
    print(f'\nTOTAL RUNTIME: {int((time.time() - start_time)/60.)} minutes {int((time.time() - start_time) % 60)} seconds')
    if not args.stdout:
        sys.stdout.close()
