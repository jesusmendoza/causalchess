"""Float64 CUDA evaluation of the standard L2 multinomial logistic objective.

Uses the same SciPy L-BFGS-B options and objective scaling as sklearn's lbfgs
LogisticRegression, moving only matrix operations to CUDA. No hyperparameter
search or test-set-dependent optimization is performed.
"""
import time
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import softmax


class GPUMultinomial:
    def __init__(self, C=1., max_iter=2000, tol=1e-4):
        self.C,self.max_iter,self.tol=C,max_iter,tol

    def fit(self, X, y):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        self.classes_,targets=np.unique(y,return_inverse=True)
        if len(self.classes_)<3:
            raise ValueError('This implementation is for multinomial (>=3 classes) fits')
        torch.backends.cuda.matmul.allow_tf32 = False
        tx=torch.as_tensor(np.ascontiguousarray(X),dtype=torch.float64,device='cuda')
        tx32=tx.float()
        ty=torch.as_tensor(targets,dtype=torch.long,device='cuda')
        shape=(len(self.classes_),X.shape[1]+1)
        reg=1./(self.C*len(y))
        calls=0; started=time.time()

        def objective(flat, warm=False):
            nonlocal calls
            dtype=torch.float32 if warm else torch.float64
            weight=torch.tensor(flat.reshape(shape),dtype=dtype,device='cuda',requires_grad=True)
            logits=torch.nn.functional.linear(tx32 if warm else tx,weight[:,:-1],weight[:,-1])
            loss=torch.nn.functional.cross_entropy(logits.double(),ty)+.5*reg*weight[:,:-1].double().square().sum()
            loss.backward()
            value=float(loss.detach().cpu())
            gradient=weight.grad.detach().double().cpu().numpy().ravel().copy()
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                raise RuntimeError('Nonfinite objective or gradient')
            calls+=1
            if calls%100==0:
                print(f'  L-BFGS warm={warm} calls={calls} loss={value:.8f} gradient_inf={abs(gradient).max():.3g} seconds={time.time()-started:.1f}',flush=True)
            return value,gradient

        # Numerical warm start only. Always finish with the exact float64
        # objective and the unchanged reference stopping criteria.
        warm=minimize(lambda w:objective(w,warm=True),np.zeros(np.prod(shape)),
                      method='L-BFGS-B',jac=True,
                      options={'maxiter':self.max_iter,'maxls':50,'gtol':self.tol,
                               'ftol':64*np.finfo(float).eps})
        result=minimize(objective,warm.x,method='L-BFGS-B',jac=True,
                        options={'maxiter':self.max_iter,'maxls':50,'gtol':self.tol,
                                 'ftol':64*np.finfo(float).eps})
        if not result.success:
            raise RuntimeError(f'Nonconverged multinomial: {result.message}')
        weights=result.x.reshape(shape)
        self.coef_=weights[:,:-1].copy()
        self.intercept_=weights[:,-1].copy()
        self.n_iter_=np.array([warm.nit+result.nit])
        self.optimization_={'objective':float(result.fun),'gradient_inf':float(abs(result.jac).max()),
                            'function_calls':int(warm.nfev+result.nfev),'status':str(result.message),
                            'warm_iterations':int(warm.nit),'final_float64_iterations':int(result.nit),
                            'dtype':'float64','backend':'SciPy L-BFGS-B / PyTorch CUDA objective'}
        return self

    def predict_proba(self,X):
        return softmax(X@self.coef_.T+self.intercept_,axis=1)
