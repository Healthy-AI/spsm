import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.extmath import safe_sparse_dot, log_logistic
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score

from scipy.special import expit
from scipy.optimize import fmin_l_bfgs_b

from .model_utils import *

def find_missing_patterns_(X, null_pattern=False, min_samples_pattern=1):
    """ Finds the unique missingness patterns in X 
    
    Args: 
        X (DataFrame): Covariates
        
    Returns: 
        DataFrame: Binary matrix representing the missing values (columns) of each pattern (rows)        
    """
    
    d = X.shape[1]
    M = 1*X.isna().drop_duplicates().values
    nM = M.sum(axis=1)
    JM = np.argsort(nM)
    M = M[JM,:]
    
    # Remove patterns with too few occurrences
    if min_samples_pattern > 1: 
        if not null_pattern:
            logging.warning('Using min_samples_pattern > 1 and null_pattern == False. Some samples may fit no pattern')
        
        J = find_closest_missing_pattern_(X, M)
        nP = np.bincount(J.ravel()) # This relies on J having 0 as smallest value and being only integers
        M = M[nP>=min_samples_pattern, :]

    # Add null pattern if doesn't exist
    if (M.shape[0]<1 or M.sum(axis=1).max() < d) and null_pattern:
        M = np.concatenate([M, np.ones((1,d))], axis=0)
    return M
    
def find_closest_missing_pattern_(X, M):
    """
    Finds the closest missingness pattern for each row in df
    
    Args: 
        X (DataFrame): Covariates
        N (DataFrame): Missingness mask
        
    Returns: 
        DataFrame: Column vector with the missing pattern ID for each row in X        
    """
    
    n, d = X.shape[0], X.shape[1]
    Xw = X.isna().values.reshape([n, d, 1])
    Mw = M.T.reshape([1, M.T.shape[0], M.T.shape[1]])
    J = np.argmin(np.abs(Xw-Mw).sum(axis=1) + (Xw > Mw).sum(axis=1)*1e6, axis=1).reshape([-1,1])
    return J


class BaseSharingLinearSubModel(BaseEstimator):
    """
    Base Sharing Pattern Submodel
    
    Base class for linear regression and logistic regression pattern submodel estimators. 
    Implements most of the optimization steps, aside from link-specific gradient steps. 
    
    Optimization inspired by https://gist.github.com/vene/fab06038a00309569cdd
    
    """
    
    def __init__(self, alpha0=0, alphap=0., tol=1e-5, fit_intercept=True, penalty_main='l2', 
                 null_pattern=True, reg_pattern_intercept=False, min_samples_pattern=1, null_use_main=True):
        """
        Constructor 
        
        Args: 
            alpha0 (float): Level of regularization of the main model
            alphap (float): Level of regularization of the pattern submodels
            tol (float): Tolerance of the L-BFGS optimizer
            fit_intercept (bool): Whether or not to fit an intercept in the models
            penalty_main (str): Penalty for main model, 'l2' or 'l1'
            null_pattern (bool): Whether or not to include a null pattern (all variables missing)
            reg_pattern_intercept (bool): Whether to regularize intercept for patterns
            min_samples_pattern (int): How many samples are required to fit parameters for a pattern
            null_use_main (bool): Whether to use the main model for samples that match no pattern 
                                  (otherwise, no variables are used for prediction)
        """
        
        self.alpha0 = alpha0
        self.alphap = alphap
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.null_pattern = null_pattern
        self.reg_pattern_intercept = reg_pattern_intercept
        self.penalty_main = penalty_main
        self.min_samples_pattern = min_samples_pattern
        self.null_use_main = null_use_main
        
    @staticmethod
    def loss_grad_(We, X, y, M, J, Jo, XOi, alpha0, alphap, link_function, fit_intercept, 
                   reg_pattern_intercept, penalty_main, null_pattern, null_use_main):
        """
        Loss and gradient in format expected by L-BFGS implementation
        """
        
        ''' 1. Get data and weights in right shape '''
        n, d = X.shape[0], X.shape[1]
        n_dim = d + 1*fit_intercept
        k = M.shape[0]
                   
        # Reshape flattened weight vector into weights for default model and patterns
        W = We[:n_dim*(k+1)].reshape([-1,n_dim]) - We[n_dim*(k+1):].reshape([-1,n_dim])
        W0 = W[:1,:]
        Wp = W[1:,:]
        
        # @TODO: regularisation of main model should depend on number of samples per variable
        ns = np.concatenate([[n], Jo.sum(axis=0)])
        if null_pattern: 
            ns[-1] = n
        
        ''' 2. Compute risk term and gradient of risk '''

        if link_function == 'linear':
            
            R = safe_sparse_dot(XOi, (W0+Wp).T) - y.reshape((-1,1))
            SE = np.square(R)
            loss = np.sum(SE*Jo, axis=0)/ns[1:] # One loss per pattern (Note: not quite empirical risk)

            # Pattern-specific gradients
            gradp = 2*safe_sparse_dot((XOi).T, R*Jo/ns[1:])  
            
        elif link_function == 'logistic': 
            
            YZ = safe_sparse_dot(XOi, (W0+Wp).T)*y 
            loss = -np.sum(log_logistic(YZ)*Jo, axis=0)/ns[1:] # One loss per pattern (Note: not quite empirical risk)
            
            
            # Pattern-specific gradients
            Z = expit(YZ)
            Z0 = (Z-1)*y
            gradp = safe_sparse_dot((XOi).T, Z0*Jo/ns[1:])
            
        else: 
            raise Exception('Unknown link function: %s. Supporting linear or logistic.' % link_function)
            
        # Overall risk    
        risk = loss.sum() # Sum loss over patterns
        
        # Main-pattern gradient
        grad0 = gradp.sum(axis=1)    
        
        # Zero out gradient for null pattern if null_use_main
        # This has to happen after the above line
        if null_pattern and null_use_main: 
            gradp[:,-1] = 0
            
        # Overall gradient for risk
        grad_risk = np.concatenate([grad0, gradp.T.ravel(), -grad0, -gradp.T.ravel()])
            
            
        ''' 3. Compute regularization terms and gradients of regularization '''

        
        # Penalty for the main pattern (l1 or l2)
        if penalty_main == 'l1':
            reg0 = alpha0*np.abs(W0[0,:d]).sum()/ns[0] 
        elif penalty_main == 'l2':
            reg0 = .5*alpha0*np.square(W0[0,:d]).sum()/ns[0] 
        else: 
            raise Exception('Unknown penalty_main: %s' % penalty_main)
        
        # Penalty for pattern-specific intercepts
        if reg_pattern_intercept:
            regp = alphap*(np.abs(Wp).sum(axis=1)/ns[1:]).sum() 
        else: 
            regp = alphap*(np.abs(Wp[:,:d]).sum(axis=1)/ns[1:]).sum() # Non-intercept part of main model

        # Gradient for main-pattern regularization
        if penalty_main == 'l1':
            grad_reg0_pos = np.array([alpha0/ns[0]]*d + [0]*(n_dim-d))
            grad_reg0_neg = grad_reg0_pos
        elif penalty_main == 'l2':
            grad_reg0_pos = np.concatenate([alpha0*W0[0,:d]/ns[0], [0]*(n_dim-d)])
            grad_reg0_neg = -grad_reg0_pos
            
        # Gradient for pattern-specific regularization
        pig = alphap if reg_pattern_intercept else 0         
        N = np.array([[ns[i]]*n_dim for i in range(1,k+1)]).ravel()
        grad_regp_pos = np.array(([alphap]*d + [pig]*(n_dim-d))*k)/N
        grad_regp_neg = grad_regp_pos
        
        # Overall gradient for regularization
        grad_reg = np.concatenate([grad_reg0_pos, grad_regp_pos, grad_reg0_neg, grad_regp_neg])
        
        ''' 4. Add objective and gradient terms ''' 
        
        obj = risk + reg0 + regp 
        grad = grad_risk + grad_reg
        
        #logging.debug('Objective: %.3f' % obj)
        
        return obj, grad
    
    def prepare_patterns_(self, X, fit_patterns=False):
        """
        Prepares data for training and prediction by identifying which missigness patterns they corresponds to. 
        If fit_patterns is set to True, the patterns used by the model are updated. 
        
        Args:
            X (array/DataFrame): Data
            fit_patterns (bool): Whether to update model missigness patterns
        """
        
        # Make sure data is dataframe
        X = get_dataframe(X)
        
        # Find unique missigness patterns
        if fit_patterns:
            M = find_missing_patterns_(X, null_pattern=self.null_pattern, min_samples_pattern=self.min_samples_pattern)
        else:
            M = self.M

        
        # Find closest patterns
        J = find_closest_missing_pattern_(X, M)
        k = M.shape[0]
        
        # One-hot
        Jo = np.zeros((J.ravel().size, k))
        Jo[np.arange(J.ravel().size), J.ravel()] = 1
        XM = np.dot(Jo, M)>0
        
        # This step is required for the null pattern behaving correctly
        # Only missing values are zero-imputed in the null pattern 
        if self.null_pattern and self.null_use_main: 
            XM[Jo[:,-1]>0,:] = np.isnan(X.values[Jo[:,-1]>0,:])
        
        # Set missing values to 0, so as to not affect prediction (gradient contrib. also zero:d for missing values)
        # @TODO: NOT SURE IF THIS STEP IS ACTUALLY NEEDED SINCE X*O IS USED IN THE GRADIENT
        X = X.values.copy()
        X.ravel()[XM.ravel()] = 0
        
        # Observed variable mask
        # @TODO: Think this is somewhat superfluous
        O = np.dot(Jo, M)<1
        
        # If null_use_main=True and null_pattern=True, the main model should be used for the null pattern
        # That means that null pattern samples contribute to updates of the main model as well
        # Whichever variables are observed, those are used and updated
        # This assumes that the data has been zero-imputed and that the null-pattern has index -1
        if self.null_pattern and self.null_use_main: 
            O[Jo[:,-1]>0,:] = 1
         
        return X, M, J, Jo, O
    
    def solve_(self, X, y, M, J, Jo, O):
        """
        Trains the model by solving the objective using L-BFGS-B
        
        Args:
            X (array/DataFrame): Inputs
            y: Outputs
            M: 
            J: 
            Jo:
            O: 
        """
        
        n = X.shape[0]
        d = X.shape[1]
        k = M.shape[0]
        
        if self.fit_intercept: 
            n_params = (d+1)*(k+1)*2
        else: 
            n_params = d*(k+1)*2
            
        # Initialize model parameters to 0 (necessary for null pattern)
        coef0 = np.zeros(n_params)
        
        # Add ones to the covariate set if intercept is being fit
        # TODO: This does not have to be done every iteration
        if self.fit_intercept:
            XOi = np.concatenate([X*O, np.ones((n, 1))], axis=1)            
        else:
            XOi = X*O
        
        # Solve objective with LBFGS-B
        coef, f, r = fmin_l_bfgs_b(self.loss_grad_, x0=coef0, fprime=None,
                                pgtol=self.tol,
                                bounds=[(0, None)] * n_params,
                                args=(X, y, M, J, Jo, XOi, self.alpha0, self.alphap, 
                                      self.fit_intercept, self.reg_pattern_intercept, 
                                      self.penalty_main, self.null_pattern, self.null_use_main))

        if self.fit_intercept:
            Coef = coef[:(d+1)*(k+1)].reshape([-1,(d+1)]) - coef[(d+1)*(k+1):].reshape([-1,(d+1)])
        else:
            Coef = coef[:d*(k+1)].reshape([-1,d]) - coef[d*(k+1):].reshape([-1,d])
           
        # Update model parameters
        self.coef0 = Coef[:1,:-1]
        self.intercept0 = Coef[0,-1]
        self.coefp = Coef[1:,:-1]
        self.interceptp = Coef[1:,-1]
        
        return Coef
    
    def set_params(self, **params):
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        
        if 'alpha0' in params:
            self.alpha0 = params['alpha0']
        if 'alphap' in params:
            self.alphap = params['alphap']
        if 'tol' in params:
            self.tol = params['tol']
        if 'fit_intercept' in params:
            self.tol = params['fit_intercept']
        if 'null_pattern' in params:
            self.tol = params['null_pattern']
        if 'reg_pattern_intercept' in params: 
            self.reg_pattern_intercept = params['reg_pattern_intercept']
        if 'penalty_main' in params: 
            self.reg_pattern_intercept = params['penalty_main']
        if 'min_samples_pattern' in params: 
            self.min_samples_pattern = params['min_samples_pattern']
        if 'null_use_main' in params: 
            self.min_samples_pattern = params['null_use_main']
            
        
        # @TODO: Change fitted status?    
        
        return self
    
    def get_params(self, deep=True):
        return {'alpha0': self.alpha0, 
                'alphap': self.alphap, 
                'tol': self.tol, 
                'fit_intercept': self.fit_intercept, 
                'null_pattern': self.null_pattern,
                'reg_pattern_intercept': self.reg_pattern_intercept,
                'penalty_main': self.penalty_main,
                'min_samples_pattern': self.min_samples_pattern,
                'null_use_main': self.null_use_main}

    def grad_check(self):
        #@TODO: Doesn't run as a function. Run by copying in front of fitting
        coef0 = np.random.rand(n_params)
        pos = np.random.rand(int(n_params/2))<.5
        zero = np.concatenate([pos, 1-pos])
        epsilon = 1e-11
        m = int(coef0.shape[0]/2)
        coef0[zero>0] = 0
        
        gs = np.zeros(m)
        for i in range(m):
            ceps = coef0.copy()
            ceps[i] += epsilon
            obj1, grad1 = self.loss_grad_(ceps, X, y, J, M, 
                                          self.alpha0, self.alphap,
                                          self.fit_intercept, self.reg_pattern_intercept, self.penalty_main,
                                          self.null_pattern, self.null_use_main)
            ceps = coef0.copy()        
            ceps[i] -= epsilon
            obj2, grad2 = self.loss_grad_(ceps, X, y, J, M, 
                                          self.alpha0, self.alphap,
                                          self.fit_intercept, self.reg_pattern_intercept, self.penalty_main,
                                          self.null_pattern, self.null_use_main)
            gs[i] = (obj1 - obj2)/(2*epsilon)
            
        print('Finite method\n', gs)
        print('Grad\n', grad1[:m])      
        
        
class SharingLogisticSubModel(BaseSharingLinearSubModel):
    """
    Sharing Pattern Submodel with Logistic Regression Estimators
    
    """

    @staticmethod
    def loss_grad_(We, X, y, M, J, Jo, XOi, alpha0, alphap, fit_intercept=True, reg_pattern_intercept=False, 
                   penalty_main='l2', null_pattern=False, null_use_main=False):
        """
        Loss and gradient in format expected by L-BFGS implementation
        """
        return BaseSharingLinearSubModel.loss_grad_(We, X, y, M, J, Jo, XOi, alpha0, alphap, 'logistic', 
                                                    fit_intercept, reg_pattern_intercept, penalty_main, 
                                                    null_pattern, null_use_main)
        
    
    def predict_proba(self, X):
        """
        Prediction probability
        """
        
        # Prepare data for prediction
        Xz, M, J, Jo, O = self.prepare_patterns_(X, fit_patterns=False)
                    
        # Compute intercept
        intercept = self.intercept0 + self.interceptp
        
        # Compute predicted probability
        Z = (safe_sparse_dot(Xz*O, (self.coef0+self.coefp).T) + intercept)*Jo
        p = expit(Z.sum(axis=1))
        
        p = np.array([1-p, p]).T

        return p

    
    def predict(self, X):
        """
        Prediction
        """
        
        p = self.predict_proba(X)

        return np.argmax(p, axis=1)
    
    
    def fit(self, X, y):
        """
        Fit model to data
        
        Args: 
            X (Numpy array or DataFrame): Covariate matrix with NaN for missing values
            y (Numpy array or DataFrame or Series): Label vector
        """
        
        # Make sure X is a dataframe for the first part of this function
        X = get_dataframe(X)
        d = X.shape[1]
        
        # Make sure Y is a numpy array with only 2 unique values and transform to -1, +1
        y = get_array(y)
        self.classes_ = sorted(np.unique(y))
        if not len(self.classes_) == 2: 
            raise Exception('Number of classes must be 2')
        y = -1*(y==y.min()) + (y==y.max())
        y = y.reshape((-1,1))
        if not y.shape[0] == X.shape[0]:
            raise Exception('Inequal number of samples in X (%d) and y (%d)' % (X.shape[0], y.shape[0]))
        
        # Find missingness patterns and prepare X for training
        X, M, J, Jo, O = self.prepare_patterns_(X, fit_patterns=True)
        self.M = M
        k = M.shape[0]
        logging.info('Found %d unique missingness patterns with at least %d samples' % (k, self.min_samples_pattern))
        
        # Solve optimization
        self.solve_(X, y, M, J, Jo, O)
        
        if self.null_pattern: 
            if not self.null_use_main:
                # Fix intercept of null pattern so that prediction without covariates matches marginal 
                # This has do be done only if no samples are observed in the null pattern during training
                if Jo[:,-1].sum()==0:
                    self.interceptp[-1] = -np.log(1./((y>0).mean())-1) - self.intercept0
            else: 
                # Predict using the main model for null patterns (deviation=0)
                pass
        
        return self
 
    def score(self, X, y):
        """
        Scores the predictions made by the model using the ROC AUC score
        """
        return roc_auc_score(y, self.predict_proba(X)[:,1])

    
    

class SharingLinearSubModel(BaseSharingLinearSubModel):
    """
    Sharing Pattern Submodel with Linear Regression Estimators
    
    Optimization inspired by https://gist.github.com/vene/fab06038a00309569cdd
    
    """
    
    @staticmethod
    def loss_grad_(We, X, y, M, J, Jo, XOi, alpha0, alphap, fit_intercept, reg_pattern_intercept, 
                   penalty_main, null_pattern, null_use_main):
        """
        Loss and gradient in format expected by L-BFGS implementation
        """
        return BaseSharingLinearSubModel.loss_grad_(We, X, y, M, J, Jo, XOi, alpha0, alphap, 'linear', 
                                                    fit_intercept, reg_pattern_intercept, penalty_main, 
                                                    null_pattern, null_use_main)
    
    def predict(self, X):
        """
        Prediction
        """
        
        # Prepare data for prediction
        Xz, M, J, Jo, O = self.prepare_patterns_(X, fit_patterns=False)
        
        # Compute intercept
        intercept = self.intercept0 + self.interceptp.reshape((1,-1))
        
        # Compute prediction
        Z = (safe_sparse_dot(Xz*O, (self.coefp + self.coef0).T) + intercept)*Jo
        yp = Z.sum(axis=1)
        
        return yp
    
    def fit(self, X, y):
        """
        Fit model to data
        
        Args: 
            X (Numpy array or DataFrame): Covariate matrix with NaN for missing values
            y (Numpy array or DataFrame or Series): Label vector
        """
        
        # Make sure X is a dataframe for the first part of this function
        X = get_dataframe(X)
        d = X.shape[1]
        
        # Make sure Y is a numpy array
        y = get_array(y)
        
        # Find missingness patterns and prepare X for training
        X, M, J, Jo, O = self.prepare_patterns_(X, fit_patterns=True)
        self.M = M
        k = M.shape[0]
        logging.info('Found %d unique missingness patterns with at least %d samples' % (k, self.min_samples_pattern))
        
        # Solve optimization
        self.solve_(X, y, M, J, Jo, O)
        
        if self.null_pattern: 
            if not self.null_use_main:
                # Fix intercept of null pattern so that prediction without covariates matches marginal 
                # This has do be done only if no samples are observed in the null pattern during training
                if Jo[:,-1].sum()==0:
                    self.interceptp[-1] = y.mean()
            else: 
                # Predict using the main model for null patterns (deviation=0)
                pass
        
        return self
 
    def score(self, X, y):
        """
        Scores the predictions made by the model using the R2 score
        """
        return r2_score(y, self.predict(X))

    