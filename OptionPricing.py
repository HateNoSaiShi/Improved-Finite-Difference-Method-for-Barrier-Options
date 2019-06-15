# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 21:34:47 2019

@author: HateNoSaiShi
"""

import numpy as np
from scipy.stats import norm
from scipy import linalg

class option(object):
    '''
        Abstract:
        --------
        All option pricing procedures are under Black-Scholes model, of which the basic assumpations are:
            constant risk free rate
            constant volatility
            constant and continuous dividend
        
        Attributes:
        ----------
            __init__():
            Black_Scholes_Call():
            Black_Scholes_Put():
            BTM_Vanilla()
            tri_bound_to_ab()
            my_dot_product()
            Projected_SOR()
            FDM_DoubleBarrier_NonUnifromGrid
    '''
 
#========
    
    def __init__(self, r, q, spot_price, strike, sig, T, option_type, exercise_type, position, T_switch = 0, **kwargs):
        ''' 
            Abstract:
            --------
            Construct an object of some option.
            
            Parameters:
            ----------
            r : risk free rate
            q : continuous dividend
            spot_price: current price of undelier
            strike: strike price
            sig: volatility
            T: time to maturity
            T_switch:
                Default/0 : T is counted in trading days, AKA 252 days a year.
                1 : T is counted in natural year days, AKA 365 days a year.
                2 : T is counted in year.
            option_type : Type of the function, e.g. Vanilla, Asian, Barrier， Lookback
            exercise_type ：Exercise type of the function, e.g. European, American
            position: call or put
            
            Types that support:
            ------------------
            option_type : exercise_type 
            Vanilla : European\American
            Knock-out-double-barrier : European/American
            Knock-in-double-barrier : European/American
        '''
        #set attibutes
        self.r = r
        self.q = q
        self.spot_price = spot_price
        self.strike = strike
        self.sig = sig
        if T_switch == 0:
            self.maturity = T / 252
        elif T_switch == 1:
            self.maturity = T / 365
        elif T_switch == 2:
            self.maturity =T
        else:
            raise ValueError('Invalid input of T_switch')
        
        #The type which are supported
        valid_OptionType = ['VANILLA', 'ASIAN', 'LOOKBACK', 'UP-AND-IN-BARRIER', 'UP-AND-OUT-BARRIER',
                            'DOWN-AND-IN-BARRIER', 'DOWN-AND-OUT-BARRIER', 'KNOCK-OUT-DOUBLE-BARRIER', 'KNOCK-IN-DOUBLE-BARRIER']
        if option_type.upper() in valid_OptionType:
            self.option_type = option_type.upper()
        else:
            raise TypeError('Currently don\'t support this type of options!')
        
        if exercise_type.upper() == 'EUROPEAN':
            self.exercise_type = 'EUROPEAN'
        elif exercise_type.upper() == 'AMERICAN':
            self.exercise_type = 'AMERICAN'
        else:
            raise TypeError('Invalid Exercise Type')
        
        #option's postion
        if position.upper() == 'CALL':
            self.position = 'CALL'
        elif position.upper() == 'PUT':
            self.position = 'PUT'
        else:
            raise TypeError('Invalid position type')
            
        #set barrier for single barrier option
        if self.option_type == 'UP-AND-OUT-BARRIER':
            if 'barrier' in kwargs.keys():
                if kwargs['barrier'] <= self.spot_price:
                    raise ValueError('Barrier should not be smaller than spot price')
                else:
                    self.barrier = kwargs['barrier']
            else:
                raise TypeError('Lack information of barrier value')
        elif self.option_type == 'UP-AND-IN-BARRIER':
            if 'barrier' in kwargs.keys():
                if kwargs['barrier'] <= self.spot_price:
                    raise ValueError('Barrier should not be smaller than spot price')
                else:
                    self.barrier = kwargs['barrier']
            else:
                raise TypeError('Lack information of barrier value')
        elif self.option_type == 'DOWN-AND-OUT-BARRIER':
            if 'barrier' in kwargs.keys():
                if kwargs['barrier'] >= self.spot_price:
                    raise ValueError('Barrier should not be larger than spot price')
                else:
                    self.barrier = kwargs['barrier']
            else:
                raise TypeError('Lack information of barrier value')
        elif self.option_type == 'DOWN-AND-IN-BARRIER':
            if 'barrier' in kwargs.keys():
                if kwargs['barrier'] >= self.spot_price:
                    raise ValueError('Barrier should not be smaller than spot price')
                else:
                    self.barrier = kwargs['barrier']
            else:
                raise TypeError('Lack information of barrier value')
        
        
        #set barriers for double barrier
        if self.option_type == 'KNOCK-OUT-DOUBLE-BARRIER':
            if 'lower_barrier' in kwargs.keys() and 'upper_barrier' in kwargs.keys():
                if kwargs['upper_barrier'] <= kwargs['lower_barrier']:
                    raise ValueError('Upper barrier should be larger than lower barrier')
                elif self.spot_price >= kwargs['upper_barrier'] or self.spot_price <= kwargs['lower_barrier']:
                    raise ValueError('Invalid barrier value')
                else:
                    self.lower_barrier = kwargs['lower_barrier']
                    self.upper_barrier = kwargs['upper_barrier']
            else:
                raise TypeError('Lack inforamtion of upper/lower barrier value')
        elif self.option_type == 'KNOCK-IN-DOUBLE-BARRIER':
            if 'lower_barrier' in kwargs.keys() and 'upper_barrier' in kwargs.keys():
                if kwargs['upper_barrier'] <= kwargs['lower_barrier']:
                    raise ValueError('Upper barrier should be larger than lower barrier')
                elif self.spot_price >= kwargs['upper_barrier'] and self.spot_price >= kwargs['lower_barrier']:
                    raise ValueError('Invalid barrier value')
                else:
                    self.lower_barrier = kwargs['lower_barrier']
                    self.upper_barrier = kwargs['upper_barrier']
            else:
                raise TypeError('Lack inforamtion of upper/lower barrier value')
#========
    
    def Black_Scholes_Call(self):
        '''
        Abstract:
        --------
        Black scholes call price
        '''
        d1 = ( np.log(self.spot_price / self.strike) + (self.r - self.q + 1 / 2 * self.sig **2 ) * self.maturity)
        d1 = d1 / (self.sig * np.sqrt(self.maturity))
        d2 = d1 - self.sig * np.sqrt(self.maturity)
        forward = self.spot_price * np.exp((self.r - self.q) * self.maturity)
        call_price = np.exp( - self.r * self.maturity) * (forward * norm.cdf(d1) - self.strike * norm.cdf(d2))
        return call_price

#========

    def Black_Scholes_Put(self):
        '''
        Abstract:
        --------
        Black scholes put price
        '''
        d1 = ( np.log(self.spot_price / self.strike) + (self.r - self.q + 1 / 2 * self.sig **2 ) * self.maturity)
        d1 = d1 / (self.sig * np.sqrt(self.maturity))
        d2 = d1 - self.sig * np.sqrt(self.maturity)
        forward = self.spot_price * np.exp((self.r - self.q) * self.maturity)
        put_price = np.exp( - self.r * self.maturity) * (self.strike * norm.cdf( - d2) - forward * norm.cdf(- d1))
        return put_price
        
#========

    def BTM_Vanilla(self, Nt):
        '''
        Abstract:
        --------
        Binomial tree pricing model for Vanilla American options
        
        Parameters:
        ----------
        Nt: Number of time periods used in BTM model
        '''
        dt = self.maturity / Nt
        #up and dowm factor
        u = np.exp(self.sig * np.sqrt(dt))
        d = 1 / u
        #up-going probability
        p = (np.exp((self.r - self.q) * dt) - d) / (u - d)
        
        #Maturity pay-off
        s_maturity = [self.spot_price * u ** (2 * i - Nt) for i in range(Nt + 1)]
        if self.position == 'CALL':
            V = [max(0, i - self.strike) for i in s_maturity]
        elif self.position == 'PUT':
            V = [max(0, self.strike - i) for i in s_maturity]
        
        #backward iteration
        for t in range(Nt - 1, -1, -1):
            #Binomial process
            V_temp = [np.exp(- self.r * dt) * (p * V[i + 1] + (1 - p) * V[i]) for i in range(len(V) - 1)]
            
            if self.exercise_type == 'AMERICAN':
                #Early exercise process
                if self.position == 'CALL':
                    V_temp = [max(self.spot_price * u ** (2 * i - t) - self.strike, V_temp[i]) for i in range(len(V_temp))]
                else:
                    V_temp = [max( - self.spot_price * u ** (2 * i - t) + self.strike, V_temp[i]) for i in range(len(V_temp))]
            
            V = V_temp
        return V[0]

#========

    @staticmethod
    def tri_bound_to_ab(coeff_m1_arr, coeff_0_arr, coeff_p1_arr):
        """
        Compute ab which stores the matrix elements of the tri-diagonal matrix bounded matrix
        ab = [
            [0, c1, ..., c_I_3, c_I_2],
            [b1, b2, ...,b_I_3, b_I_1],
            [a2, a3, ...,a_I_1, 0]
        ]
        """
        length = coeff_m1_arr.size
        ab = np.zeros((3, length))
        ab[0, 1:length] = coeff_p1_arr[0:length - 1]
        ab[1, :] = coeff_0_arr[:]
        ab[2, 0:length - 1] = coeff_m1_arr[1:]
        return ab

#========
        
    @staticmethod
    def my_dot_product(array_a, array_b, array_c, vector):
        n = array_a.size
        a = np.zeros(n)
        c = np.zeros(n)
        a[1:] = array_a[1:] * vector[0:-1]
        b = array_b * vector
        c[0:-1] = array_c[0:-1] * vector[1:]

        if not np.isfinite(b).all():
            pass
        array_result = np.array(a + b + c)
        return array_result

#========

    def Projected_SOR(self, array_a, array_b, array_c, b, x_0, payoff, t, step, s):
        '''
        Abstract:
        --------
        Projected SOR method. Use for prcing American Options under Finite Difference Framework
        
        Parameters:
        ----------
        array_a : -1 diagonal
        array_b : diagonal
        array_c : 1 diagonal
        '''
        #print(len(array_b))
        A = np.diag(array_a, -1) + np.diag(array_b, 0) + np.diag(array_c, 1)
        length = len(array_b)
        epsilon = 0.01 * length
        omega = 0.5
        x_k = x_0.copy()
        while True:
            x_k_plus = [] 
            for i in range(length):
                if i == 0:
                    sum_1 = 0
                else:
                    sum_1 = A[i][i - 1] * x_k_plus[i - 1]
                if i < length - 1:
                    sum_2 = A[i][i + 1] * x_k[i + 1]          
                else:
                    sum_2 = 0
                x_k_plus_gs_temp = (-sum_1 - sum_2 + b[i]) / A[i][i]
                x_k_plus_temp = max((1 - omega) * x_k[i] + omega * x_k_plus_gs_temp, payoff[i])
                x_k_plus.append(x_k_plus_temp)
            if self.option_type == 'KNOCK-OUT-DOUBLE-BARRIER' or self.option_type == 'KNOCK-IN-DOUBLE-BARRIER':
                if t % step == 0:
                    x_k_plus *= np.where(
                        (s <= self.upper_barrier) & (s >= self.lower_barrier), 1, 0)
            if self.option_type == 'DOWN-AND-OUT-BARRIER' or self.option_type == 'DOWN-AND-IN-BARRIER':
                if t % step == 0:
                    x_k_plus *= np.where(s >= self.barrier, 1, 0)
            if self.option_type == 'UP-AND-OUT-BARRIER' or self.option_type == 'UP-AND-IN-BARRIER':
                if t % step == 0:
                    x_k_plus *= np.where(s <= self.barrier, 1, 0)
            if np.sqrt(sum((np.abs(np.array(x_k_plus) - np.array(x_k))) ** 2)) < epsilon:
                break
            else:
                x_k = x_k_plus.copy()
        return x_k_plus
        
#========

    def FDM_DoubleBarrier_NonUnifromGrid(self, Ns, Nt, theta, ratio, m):
        '''
        Abstract:
        --------
        Finite difference method for double barrier option.
        Using a non-uniform grid, with shape controlled by input ratio.
        Discrete monioring.
        Iteration process is only suitable for double knock out option.
        For double knock in, this funcition uses a vanilla option to minus the identical knock out.
        
        Parameters:
        ----------
        Ns: Number of points in price axis
        Nt: Number of points in time axis
        theta:
            0 : Fully implicit method
            0.5 : Crank-nicolson method
            According to reference, fully implicit method is better than crank-nicolson method
            Reference :
            Zvan R, Vetzal K R, Forsyth P A. PDE methods for pricing barrier options ☆[J]. 
            Journal of Economic Dynamics & Control, 1997, 24(11-12):1563-1590.      
        ratio: The parameter use to controll the shape of the grid
        m : monitoring times
        '''
        # discretize Nt-1 points between every two monitoring time, total Nt*m + 1 gird in time axis
        step = Nt
        Nt = Nt * m

        # set up parameters
        mu = self.r - self.q
        _range = 5 * self.sig * np.sqrt(self.maturity)
        Smax = max(self.upper_barrier, max(self.spot_price, self.strike) * np.exp((mu - self.sig ** 2 / 2.0) * self.maturity + _range)) * 1.0000001
        Smin = self.lower_barrier * 0.9999999
        # totally Nt + 1 in row grid
        dt = self.maturity/ float(Nt)  

        # generate non-uniform grid
        s = np.linspace(Smin, Smax, Ns * (1 - ratio) + 1)
        temp = [self.lower_barrier, self.spot_price, self.upper_barrier]
        lower_index = np.array([sum(s < i) for i in temp]) -1
        upper_index = lower_index + 1
        delta_s = - s[lower_index] + s[upper_index]
        delta_s = delta_s[0]
        if lower_index[1] > lower_index[0] and lower_index[2] > lower_index[1]:        
            count = int(Ns * ratio / 3.0)
        else:
            count = int(Ns * ratio / 2.0)
        ds = delta_s / (count - 1)
        
        #enforce the grid density around key value
        insert_vector = [np.linspace(s[lower_index[j]] + ds, s[upper_index[j]] - ds, count ) for j in [0, 1, 2]]
        s_temp = np.append(s[:lower_index[0]+1],insert_vector[0])
        s_temp = np.append(s_temp,s[upper_index[0]:lower_index[1]+1])       
        if lower_index[1] > lower_index[0]:        
            s_temp = np.append(s_temp,insert_vector[1])
        s_temp = np.append(s_temp,s[upper_index[1]:lower_index[2]+1])
        if lower_index[2] > lower_index[1]:
            s_temp = np.append(s_temp,insert_vector[2])
        s_temp = np.append(s_temp,s[upper_index[2]:])
        s = s_temp
        Ns = len(s) - 1

        # initialize the payoff
        if self.position == 'CALL':
            V_Nt = np.maximum(s - self.strike, 0) * np.where(
                    (s <= self.upper_barrier) & (s >= self.lower_barrier), 1, 0)
            payoff = np.maximum(s - self.strike, 0)
        else:
            V_Nt = np.maximum(self.strike - s, 0) * np.where(
                    (s <= self.upper_barrier) & (s >= self.lower_barrier), 1, 0)
            payoff = np.maximum(- s + self.strike, 0)
        
        # initialize the Dirichlet boundary condition
        if self.position == "CALL":
            f_0 = np.linspace(0, 0, Nt + 1)
            f_Ns = np.linspace(0, 0, Nt + 1)
        elif self.position == "PUT":
            f_0 = np.linspace(0, 0, Nt + 1)
            f_Ns = np.linspace(0, 0, Nt + 1)

        # initialize the tridiagonal matrix by scalar-form
        delta_s_i = 0.5 * (s[2:] - s[0:Ns - 1])
        delta_s_plus =  s[2:] - s[1:Ns] 
        delta_s_minus = s[1:Ns] - s[0:Ns - 1]
 
        
        # from a_2 to a_I-1 are in the calculation matrix
        a = - (1.0 - theta) * self.sig **2 * s[1:Ns] **2 / (2.0 * delta_s_i * delta_s_minus) + (
            1 - theta) * mu * s[1:Ns] / (2 * delta_s_i)
        # from b_1 to b_I-1 are in the calculation matrix
        b =  1.0 / dt + (1 - theta) * self.r + (1.0 - theta) * self.sig **2 * s[1:Ns] **2 / (2.0 * delta_s_i * delta_s_minus)
        b = b + (1.0 - theta) * self.sig **2 * s[1:Ns] **2 / (2.0 * delta_s_i * delta_s_plus)
        # from c_1 to c_I-2 are in the calculation matrix
        c = - (1.0 - theta) * self.sig **2 * s[1:Ns] **2 / (2.0 * delta_s_i * delta_s_plus) - (
            1 - theta) * mu * s[1:Ns] / (2 * delta_s_i)
        # from alpha_2 to alpha_I-1 are in the calculation matrix
        alpha = theta * self.sig **2 * s[1:Ns] **2 / (2.0 * delta_s_i * delta_s_minus
            ) - theta * mu * s[1:Ns] / (2 * delta_s_i)
        # from beta_1 to beta_I-1 are in the calculation matrix
        beta =  1.0 / dt - theta * self.sig **2 * s[1:Ns] **2 / (2.0 * delta_s_i * delta_s_minus) - self.r * theta
        beta = beta - theta * self.sig **2 * s[1:Ns] **2 / (2.0 * delta_s_i * delta_s_plus)
        # from gamma_1 to gamma_I-2 are in the calculation matrix
        gamma = theta * self.sig **2 * s[1:Ns] **2 / (2.0 * delta_s_i * delta_s_plus) + theta * mu * s[1:Ns] / (
            2 * delta_s_i)
        
        # From Nt to 1, calculate V_Nt-1, V_Nt-2, ..., V_0 (vectors)
        V_Nplus = V_Nt[1:Ns]
        for k in range(Nt, 0, -1):
        #for k in range(1,0,-1):
            #V_Nplus : b of Ax=b
            V_Nplus = self.my_dot_product(alpha, beta, gamma, V_Nplus)
            V_Nplus[0] = V_Nplus[0] - a[0] * f_0[k-1] + alpha[0] * f_0[k]
            V_Nplus[Ns-2] = V_Nplus[Ns-2] - c[Ns-2] * f_Ns[k-1] + gamma[Ns-2] * f_Ns[k]
            
            #V_N : Intial Guess for american case / x of Ax=b for european case
            ab = self.tri_bound_to_ab(a,b,c)
            V_N = linalg.solve_banded((1, 1), ab, V_Nplus)
            
            #American process
            if self.exercise_type == 'AMERICAN':
                V_N = self.Projected_SOR(a[1:],b,c[:-1], V_Nplus, V_N, payoff[1:-1], k, step, s[1:Ns])
            V_Nplus = V_N
            
            #Monitoring process
            if k % step == 0:
                V_Nplus = V_Nplus * np.where(
                    (s[1:Ns] <= self.upper_barrier) & (s[1:Ns] >= self.lower_barrier), 1, 0)
                
        # linear interpolation
        index = sum(s < self.spot_price)
        w = (self.spot_price - s[index-1]) / (s[index] - s[index-1])
        v_0 = V_Nplus[index-1] * (1 - w) + w * V_Nplus[index]

        '''
        Above process is only for double knock out option
        '''
        
        if self.option_type == 'KNOCK-OUT-DOUBLE-BARRIER':
            return v_0
        else:
            if self.position == 'CALL':
                v_0 = self.Black_Scholes_Call() - v_0
                return v_0
            else:
                if self.exercise_type == 'EUROPEAN':
                    v_0 = self.Black_Scholes_Put() - v_0
                    return v_0
                else:
                    v_0 = self.BTM_Vanilla(1200) - v_0
                    return v_0
        
#========

    def FDM_SingleBarrier_NonUnifromGrid(self, Ns, Nt, theta, ratio, m):
        '''
        Abstract:
        --------
        Finite difference method for barrier option.
        Using a non-uniform grid, with shape controlled by input ratio.
        Discrete monioring.
        Iteration process is only suitable for knock out option.
        For knock in, this funcition uses a vanilla option to minus the identical knock out.
        
        Parameters:
        ----------
        Ns: Number of points in price axis
        Nt: Number of points in time axis
        theta:
            0 : Fully implicit method
            0.5 : Crank-nicolson method
            According to reference, fully implicit method is better than crank-nicolson method
            Reference :
            Zvan R, Vetzal K R, Forsyth P A. PDE methods for pricing barrier options ☆[J]. 
            Journal of Economic Dynamics & Control, 1997, 24(11-12):1563-1590.        
        ratio: The parameter use to controll the shape of the grid
        m : monitoring times
        '''
        # discretize Nt-1 points between every two monitoring time, total Nt*m + 1 gird in time axis
        step = Nt
        Nt = Nt * m

        # set up parameters
        mu = self.r - self.q
        _range = 5 * self.sig * np.sqrt(self.maturity)
        
        if self.option_type == 'DOWN-AND-OUT-BARRIER' or self.option_type == 'DOWN-AND-IN-BARRIER':
            Smax = self.spot_price * np.exp((mu - self.sig ** 2 / 2.0) * self.maturity + _range)
            Smin = self.barrier * 0.99999999
        elif self.option_type == 'UP-AND-OUT-BARRIER' or self.option_type == 'UP-AND-IN-BARRIER':
            Smax = max(self.barrier, self.strike) * 1.0000001
            Smin = 0
        
        
        # totally Nt + 1 in row grid
        dt = self.maturity/ float(Nt)  

        
        # generate non-uniform grid
        s = np.linspace(Smin, Smax, Ns * (1 - ratio) + 1)
        if self.option_type == 'DOWN-AND-OUT-BARRIER':
            temp = [self.barrier, self.spot_price]
        elif self.option_type == 'UP-AND-OUT-BARRIER':
            temp = [self.spot_price, self.barrier]
        elif self.option_type == 'DOWN-AND-IN-BARRIER':
            temp = [self.barrier, self.spot_price]
        else:
            temp = [self.spot_price, self.barrier]
        lower_index = np.array([sum(s < i) for i in temp]) -1
        upper_index = lower_index + 1
        delta_s = - s[lower_index] + s[upper_index]
        delta_s = delta_s[0]
        if lower_index[1] > lower_index[0]:        
            count = int(Ns * ratio / 2.0)
        else:
            count = int(Ns * ratio)
        ds = delta_s / (count - 1)
        
        #enforce the grid density around key value
        insert_vector = [np.linspace(s[lower_index[j]] + ds, s[upper_index[j]] - ds, count ) for j in [0, 1]]
        s_temp = np.append(s[:lower_index[0]+1],insert_vector[0])
        s_temp = np.append(s_temp,s[upper_index[0]:lower_index[1]+1])       
        if lower_index[1] > lower_index[0]:        
            s_temp = np.append(s_temp,insert_vector[1])
        s_temp = np.append(s_temp,s[upper_index[1]:])
        s = s_temp
        Ns = len(s) - 1
        
        # initialize the payoff
        if self.position == 'CALL':
            if self.option_type == 'DOWN-AND-OUT-BARRIER' or self.option_type == 'DOWN-AND-IN-BARRIER':
                V_Nt = np.maximum(s - self.strike, 0) * np.where(s >= self.barrier, 1, 0)
            else:
                V_Nt = np.maximum(s - self.strike, 0) * np.where(s <= self.barrier, 1, 0)
            payoff = np.maximum(s - self.strike, 0)
        elif self.position == 'PUT':
            if self.option_type == 'DOWN-AND-OUT-BARRIER' or self.option_type == 'DOWN-AND-IN-BARRIER':
                V_Nt = np.maximum(- s + self.strike, 0) * np.where(s >= self.barrier, 1, 0)
            else:
                V_Nt = np.maximum(- s + self.strike, 0) * np.where(s <= self.barrier, 1, 0)
            payoff = np.maximum(- s + self.strike, 0)
        
        # initialize the Dirichlet boundary condition
        if self.position == "CALL":
            f_0 = np.linspace(0, 0, Nt + 1)
            if self.option_type == 'DOWN-AND-OUT-BARRIER' or self.option_type == 'DOWN-AND-IN-BARRIER':
                f_Ns = Smax * np.exp(-self.r * np.linspace(0, self.maturity, Nt + 1)
                        ) - self.strike * np.exp(-self.r * (np.linspace(0, self.maturity, Nt + 1)))
            else:
                f_Ns = np.linspace(0, 0, Nt + 1)
        elif self.position == "PUT":
            if self.option_type == 'DOWN-AND-OUT-BARRIER' or self.option_type == 'DOWN-AND-IN-BARRIER':
                f_0 = np.linspace(0, 0, Nt + 1)
            else:
                f_0 = self.strike * np.exp(-self.r * np.linspace(0, self.maturity, Nt + 1))
            f_Ns = np.linspace(0, 0, Nt + 1)

        # initialize the tridiagonal matrix by scalar-form
        delta_s_i = 0.5 * (s[2:] - s[0:Ns - 1])
        delta_s_plus =  s[2:] - s[1:Ns] 
        delta_s_minus = s[1:Ns] - s[0:Ns - 1]
 
        
        # from a_2 to a_I-1 are in the calculation matrix
        a = - (1.0 - theta) * self.sig **2 * s[1:Ns] **2 / (2.0 * delta_s_i * delta_s_minus) + (
            1 - theta) * mu * s[1:Ns] / (2 * delta_s_i)
        # from b_1 to b_I-1 are in the calculation matrix
        b =  1.0 / dt + (1 - theta) * self.r + (1.0 - theta) * self.sig **2 * s[1:Ns] **2 / (2.0 * delta_s_i * delta_s_minus)
        b = b + (1.0 - theta) * self.sig **2 * s[1:Ns] **2 / (2.0 * delta_s_i * delta_s_plus)
        # from c_1 to c_I-2 are in the calculation matrix
        c = - (1.0 - theta) * self.sig **2 * s[1:Ns] **2 / (2.0 * delta_s_i * delta_s_plus) - (
            1 - theta) * mu * s[1:Ns] / (2 * delta_s_i)
        # from alpha_2 to alpha_I-1 are in the calculation matrix
        alpha = theta * self.sig **2 * s[1:Ns] **2 / (2.0 * delta_s_i * delta_s_minus
            ) - theta * mu * s[1:Ns] / (2 * delta_s_i)
        # from beta_1 to beta_I-1 are in the calculation matrix
        beta =  1.0 / dt - theta * self.sig **2 * s[1:Ns] **2 / (2.0 * delta_s_i * delta_s_minus) - self.r * theta
        beta = beta - theta * self.sig **2 * s[1:Ns] **2 / (2.0 * delta_s_i * delta_s_plus)
        # from gamma_1 to gamma_I-2 are in the calculation matrix
        gamma = theta * self.sig **2 * s[1:Ns] **2 / (2.0 * delta_s_i * delta_s_plus) + theta * mu * s[1:Ns] / (
            2 * delta_s_i)
        
        # From Nt to 1, calculate V_Nt-1, V_Nt-2, ..., V_0 (vectors)
        V_Nplus = V_Nt[1:Ns]
        for k in range(Nt, 0, -1):
        #for k in range(1,0,-1):
            #V_Nplus : b of Ax=b
            V_Nplus = self.my_dot_product(alpha, beta, gamma, V_Nplus)
            V_Nplus[0] = V_Nplus[0] - a[0] * f_0[k-1] + alpha[0] * f_0[k]
            V_Nplus[Ns-2] = V_Nplus[Ns-2] - c[Ns-2] * f_Ns[k-1] + gamma[Ns-2] * f_Ns[k]
            
            #V_N : Intial Guess for american case / x of Ax=b for european case
            ab = self.tri_bound_to_ab(a,b,c)
            V_N = linalg.solve_banded((1, 1), ab, V_Nplus)
            
            #American process
            if self.exercise_type == 'AMERICAN':
                V_N = self.Projected_SOR(a[1:],b,c[:-1], V_Nplus, V_N, payoff[1:-1], k, step, s[1:Ns])
            V_Nplus = V_N
            
        # linear interpolation
        index = sum(s < self.spot_price)
        w = (self.spot_price - s[index-1]) / (s[index] - s[index-1])
        v_0 = V_Nplus[index-1] * (1 - w) + w * V_Nplus[index]

        '''
        Above process is only for knock out option
        '''
        
        if self.option_type == 'UP-AND-OUT-BARRIER' or self.option_type == 'DOWN-AND-OUT-BARRIER':
            return v_0
        else:
            if self.position == 'CALL':
                v_0 = self.Black_Scholes_Call() - v_0
                return v_0
            else:
                if self.exercise_type == 'EUROPEAN':
                    v_0 = self.Black_Scholes_Put() - v_0
                    return v_0
                else:
                    v_0 = self.BTM_Vanilla(1200) - v_0
                    return v_0
        
#========

    def FDM_Vanilla_Implicit(self, Ns, Nt, m):
        '''
        Abstract:
        --------
        Finite difference method for vanilla option.
        Trivial implicit method.
               
        Parameters:
        ----------
        Ns: Number of points in price axis
        Nt: Number of points in time axis
        m : monitoring times
        '''
        
        # discretize Nt-1 points between every two monitoring time, total Nt*m + 1 gird in time axis
        step = Nt
        Nt = Nt * m

        # set up parameters
        mu = self.r - self.q
        _range = 3 * self.sig * np.sqrt(self.maturity)
        Smax = self.spot_price * np.exp((mu - self.sig ** 2 / 2.0) * self.maturity + _range)
        Smin = 0
        # totally Nt + 1 in row grid
        dt = self.maturity/ float(Nt)  
        ds = (Smax - Smin) / float(Ns)  # totally Ns + 1 in column grid
        
        # initialize the payoff
        sGrid = np.linspace(Smin, Smax, Ns + 1)
        
        if self.position.upper() == "CALL":
            V_Nt = np.maximum(sGrid - self.strike, 0)
        elif self.position.upper() == "PUT":
            V_Nt = np.maximum(self.strike - sGrid, 0)
            
        s = np.linspace(Smin, Smax, Ns + 1)
        payoff = np.maximum(s - self.strike, 0)
                
        # initialize the Dirichlet boundary condition
        if self.position.upper() == "CALL":
            f_0 = np.linspace(0, 0, Nt + 1)
            f_Ns = Smax * np.exp(-self.q * np.linspace(0, self.maturity, Nt + 1)
                                  ) - self.strike * np.exp(-self.r * (np.linspace(0, self.maturity, Nt + 1)))
        elif self.position.upper() == "PUT":
            f_0 = self.strike * np.exp(-self.r * np.linspace(0, self.maturity, Nt + 1))
            f_Ns = np.linspace(0, 0, Nt + 1)
        else:
            raise ValueError('Invalid option_type!!')

        # initialize the tridiagonal matrix by scalar-form
        i = np.linspace(1, Ns - 1, Ns - 1)
        # from a_2 to a_I-1 are in the calculation matrix
        a = -(self.sig ** 2 * i ** 2 - (self.r - self.q) * i) * dt / 2.0
        # from b_1 to b_I-1 are in the calculation matrix
        b = 1 + self.sig ** 2 * i ** 2 * dt + self.r * dt
        # from c_1 to c_I-2 are in the calculation matrix
        c = -(self.sig ** 2 * i ** 2 + (self.r - self.q) * i) * dt / 2.0

        # From Nt to 1, calculate V_Nt-1, V_Nt-2, ..., V_0 (vectors)
        V_Nplus = V_Nt[1:Ns]
        for k in range(Nt, 0, -1):
            V_Nplus[0] = V_Nplus[0] - a[0] * f_0[k]
            V_Nplus[Ns-2] = V_Nplus[Ns-2] - c[Ns-2] * f_Ns[k]
            ab = self.tri_bound_to_ab(a,b,c)
            V_N = linalg.solve_banded((1, 1), ab, V_Nplus)
            #American process
            if self.exercise_type == 'AMERICAN':
                V_N = self.Projected_SOR(a[1:],b,c[:-1], V_Nplus, V_N, payoff[1:-1], k, step, s[1:Ns])
            V_Nplus = V_N

        # linear interpolation
        w = (self.spot_price - sGrid[int(self.spot_price/ds)]) / (sGrid[int(self.spot_price/ds) + 1] - sGrid[int(self.spot_price/ds)])
        v_0 = V_N[int(self.spot_price/ds)] * (1 - w) + w * V_N[int(self.spot_price/ds) + 1]

        return v_0       
        
#========

    def Monte_Carlo_Vanilla(self, path_num):
        '''
        Abstract:
        --------
        Monte Carlo method for European vanilla option.
        
        Parameter：
        ---------
        path_num : number of simulation times
        
        '''
        mu = self.r - self.q
        simulated_underlier_price_list = [self.spot_price * np.exp((mu - 0.5 * self.sig ** 2
                                ) * self.maturity + self.sig * np.sqrt(self.maturity) * np.random.normal(0, 1, 1)) for i in range(path_num)]
        simulated_underlier_price_list = [item[0] for item in simulated_underlier_price_list]
        if self.position == 'CALL':
            simulated_option_price_list = [max(temp_price - self.strike, 0) for temp_price in simulated_underlier_price_list]
        else:
            simulated_option_price_list = [max( - temp_price + self.strike, 0) for temp_price in simulated_underlier_price_list]
        expectation = sum(simulated_option_price_list) / len(simulated_option_price_list) * np.exp(-self.r * self.maturity)
        return expectation
        
#=====================================================================================

def main(option_position):
    '''
    Show pricing result of various method
    '''
    
    r = 0.05
    q = 0
    spot_price = 100
    sig = 0.2
    T = 1
    T_switch = 2
    strike = 90
    option_type = 'down-and-out-barrier'
    exercise_type = 'european'
    position = option_position
    
    analytical_price_list = []
    trivial_pde_price_list = []
    improved_pde_price_list = []
    mc_price_list = []
    strike_list = [80 + i for i in range(41)]
                   
    print('Test for %s options' % option_position)   
    print(' ')            
    
    for strike in strike_list:
        test_option = option(r, q, spot_price, strike, sig, T, option_type, exercise_type, position,T_switch, barrier = 0.000001)
        
        '''
        analytical result
        '''
        if test_option.position == 'CALL':
            analytical_price_list.append(test_option.Black_Scholes_Call())
        else:
            analytical_price_list.append(test_option.Black_Scholes_Put())
        '''
        trivial pde result
        '''
        trivial_pde_price_list.append(test_option.FDM_Vanilla_Implicit(1200, 24, 50))
        
        '''
        improved pde result
        '''
        improved_pde_price_list.append(test_option.FDM_SingleBarrier_NonUnifromGrid(1200, 24, 0, 0.15, 50))
        
        '''
        mc result
        '''
        mc_price_list.append(test_option.Monte_Carlo_Vanilla(1200))
        
    print('strike        analytical          trivial_pde        improved_pde      mc')
    for i in range(41):
        print(str(strike_list[i]) + '            ' +str(analytical_price_list[i]) + '     ' + str(trivial_pde_price_list[i]) +
              '     ' + str(improved_pde_price_list[i]) + '     ' + str(mc_price_list[i]))
    print('')
    print('------------over----------')
    print('')
    
    
#====================================================================================

def ConvergenceAnalysis(option_position):
    '''
    Analyze convergence speed of pde method and monte carlo method
    '''
    r = 0.05
    q = 0
    spot_price = 100
    sig = 0.2
    T = 1
    T_switch = 2
    strike = 90
    option_type = 'down-and-out-barrier'
    exercise_type = 'european'
    position = option_position
    
    strike_list = [80 + i for i in range(41)]
    
    improved_pde_error_dic = {}
    mc_error_dic = {}
    
    for strike in strike_list:
        test_option = option(r, q, spot_price, strike, sig, T, option_type, exercise_type, position,T_switch, barrier = 0.000001)
        
        compare_result1 = test_option.FDM_SingleBarrier_NonUnifromGrid(400, 8, 0, 0.15, 50)
        compare_result2 = test_option.Monte_Carlo_Vanilla(50)
        para1,para2 = 400, 8
        for i in range(5):
            para1 *= 2
            para2 *= 2
            improved_pde_result = test_option.FDM_SingleBarrier_NonUnifromGrid(para1, para2, 0, 0.15, 50)
            mc_result = test_option.Monte_Carlo_Vanilla(para1)

            improved_pde_error_dic[(strike,para1)] = abs(improved_pde_result - compare_result1)
            mc_error_dic[(strike,para1)] = abs(mc_result - compare_result2)

            compare_result1 = improved_pde_result
            compare_result2 = mc_result

        print(strike)
    return pde_error_dic,mc_error_dic
    
#=====================================================================================
if __name__ == '__main__':
    main('call')
    main('put')
    #pde_error_dic,mc_error_dic = ConvergenceAnalysis('call')
