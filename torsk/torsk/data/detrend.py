import numpy as np
import scipy.linalg as la
from torsk.data.utils import upscale, downscale
from scipy.fftpack import dctn, idctn

def polynomial_trend(ft,d):
    """Finds best least-squares 'd'-degree polynomial fitting the series 'ft'."""
    n = len(ft)
    xs = np.arange(n);
    B  = np.array([xs**j for j in range(d+1)]).T
    return la.lstsq(B, ft.reshape(n,1))[0];
    
def cycles(ft,cycle_length):
    """Separetes the series 'ft' into a list cycles of length 'cycle_length'."""
    n = len(ft);
    n_cycles  = n//cycle_length;
    nC = n_cycles*cycle_length;
    return ft[:nC].reshape(n_cycles,cycle_length), ft[nC:];


def separate_trend_scaled(ft,nT,Cycle_length):
    """Separate out the quadratic trend and average cycle from a time series 'ft'.

    Given a time scale nT for which the cycle length is an integer 'Cycle_length', 
    this computes and removes the quadratic trend, then scales 'ft' smoothly to 'nT' 
    points, computes the average cycle, and removes it from the data, and scales 
    back to the original time scale.

    Returns: (ft_detrended,b,C), where b=(b0,b1,b2) are the coefficients of the 
             quadratic trend, and C is the average cycle.

    See also: recombine_trend_scaled(), the inverse of this operation.
    """

    nt = len(ft);
    
    # Extract the quadratic trend
    b=polynomial_trend(ft,2);
    
    # Remove trend
    ts=np.arange(nt);
    trend=b[0]+b[1]*ts+b[2]*ts*ts;
    
    # Remove average cycle
    fT=upscale(ft-trend,nT);
    fC,fr=cycles(fT,Cycle_length);
    C =np.mean(fC,axis=0);
    
    fT_detrended=np.concatenate([
        (fC-C).reshape((-1,)),
        fr-C[:len(fr)]
    ]);

    ft_detrended = downscale(fT_detrended,nt)
    
    return (ft_detrended,b,C);

def recombine_trend_scaled(ft,b,C,nT):
    """Recombine the quadratic trend and average cycle with a de-trended time series.
    
    Given the output of separate_trend_scaled() together with the time scale 'nT',
    this re-combines them into the original time series.
    """    
    nt=len(ft);
    Cycle_length=len(C);
    
    # Add back quadratic trend
    ts=np.arange(nt);
    trend=b[0]+b[1]*ts+b[2]*ts*ts;    
    
    # Add back average cycle 
    fT=upscale(ft+trend,nT);
    fC,fr=cycles(fT,Cycle_length);    
    
    fT_retrended=np.concatenate([
        (fC+C).reshape((-1,)),
        fr+C[:len(fr)]
    ]);
    
    ft_retrended=downscale(fT_retrended,nt);
    
    return ft_retrended;

def separate_trends_scaled(Ftkk,nT,Cycle_length):
    (nt,nk1,nk2)=Ftkk.shape;
    
    bkk=np.empty((nk1,nk2,3));
    Ckk=np.empty((nk1,nk2,Cycle_length));
    Ftkk_detrended=np.empty(Ftkk.shape);

    # TODO: use vectorized numpy
    for k1 in range(nk1):
        for k2 in range(nk2):
            (ft,b,C) = separate_trend_scaled(Ftkk[:,k1,k2],nT,Cycle_length);
            Ckk[k1,k2]=C;
            bkk[k1,k2]=b.flatten();
            Ftkk_detrended[:,k1,k2]=ft;
    return (Ftkk_detrended,bkk,Ckk)


def recombine_trends(Ftkk_detrended,bkk,Ckk,nT,Cycle_length):
    (nt,nk1,nk2)=Ftkk_detrended.shape;
    Ftkk=np.empty((nt,nk1,nk2));
    
    #TODO: use vectorized numpy
    for k1 in range(nk1):
        for k2 in range(nk2):
            Ftkk[:,k1,k2] = recombine_trend_scaled(
                Ftkk_detrended[:,k1,k2],bkk[k1,k2],Ckk[k1,k2],nT
            )
    return Ftkk;

# Predict from starting point and calculated trend+avg. cycle.
# Assumes that prediction starts at avg. cycle start
def kspace_predict_from_trend(f0kk, trend,  t0, pred_length):
    (bkk,Ckk,nT,Cycle_length) = trend;

    # Quadratic trend for prediction-times. Note: trend is in original time-scale
    ts = np.arange(t0,t0+pred_length);    
    quadratic_trend=bkk[0]+bkk[1]*ts+bkk[2]*ts*ts;

    # Get new coefficients for trend, shifting time-scale to t0=0
    (nk1,nk2) = f0kk.shape;
    bkk_new=np.array([[polynomial_trend(quadratic_trend[i,j],2) for j in range(nk2)] for i in range(nk1)])

    
    one = np.ones(pred_length); # Set Ftkk_detrended to constant f0kk for all t.
    Ftkk_detrended = f0kk[None,:,:]*one[:,None,None];
    Ftkk_retrended = recombine_trends(Ftkk_detrended,bkk_new,Ckk,nT,Cycle_length)

    return Ftkk_retrended;


def predict_from_trend_scaled(training_Ftxx,nT,Cycle_length, pred_length):
    (nk1,nk2)     = training_Ftxx.shape[1:]; # For now
    training_Ftkk = dct2(Ftxx,nk1,nk2);

    (Ftkk_detrended,bkk,Ckk) =  separate_trends(training_Ftkk,nT,Cycle_length);

    t0                = Ftkk.shape[0];      # Where we start predicting from
    training_ts       = np.arange(0,t0);    # ts corresponding to training    
    new_nT            = int((nT/t0)*pred_length); # pred_xlength in time-scale where cycle has integer length
    
    trend_descriptors = (bkk,Ckk,new_nT,Cycle_length);
    f0kk              = Ftkk_detrended[-1];  # We start where the training data ends

    predicted_Ftkk = kspace_predict_from_trend(f0kk,trend_descriptors, t0, pred_length);



def separate_trend_unscaled(ft,cycle_length):
    nt = len(ft);
    
    # Extract the quadratic trend
    b=polynomial_trend(ft,2);
    
    # Remove trend
    ts=np.arange(nt);
    trend=b[0]+b[1]*ts+b[2]*ts*ts;
    
    # Remove average cycle
    fc,fr=cycles(ft-trend,cycle_length);
    c =np.mean(fc,axis=0);
    
    ft_detrended=np.concatenate([
        (fc-c).reshape((-1,)),
        fr-c[:len(fr)]
    ]);

    return (ft_detrended,b,c);
    
    
def separate_trends_unscaled(Ftkk,cycle_length):
    (nt,nk1,nk2)=Ftkk.shape;
    
    bkk=np.empty((nk1,nk2,3));
    Ckk=np.empty((nk1,nk2,cycle_length));
    Ftkk_detrended=np.empty(Ftkk.shape);

    # TODO: use vectorized numpy
    for k1 in range(nk1):
        for k2 in range(nk2):
            (ft,b,C) = separate_trend_unscaled(Ftkk[:,k1,k2],cycle_length);
            Ckk[k1,k2]=C;
            bkk[k1,k2]=b.flatten();
            Ftkk_detrended[:,k1,k2]=ft;
    return (Ftkk_detrended,bkk,Ckk)
    

def recombine_trend_unscaled(ft,b,C):
    """Recombine the quadratic trend and average cycle with a de-trended time series.
    
    Given the output of separate_trend_scaled() together with the time scale 'nT',
    this re-combines them into the original time series.
    """    
    nt=len(ft);
    cycle_length=len(C);
    
    # Add back quadratic trend
    ts=np.arange(nt);
    trend=b[0]+b[1]*ts+b[2]*ts*ts;    
    
    # Add back average cycle 
    fC,fr=cycles(ft,cycle_length);    
    
    ft_retrended=np.concatenate([
        (fC+C).reshape((-1,)),
        fr+C[:len(fr)]
    ]);
    
    return ft_retrended;


def recombine_trends_unscaled(Ftkk_detrended,bkk,Ckk):
    (nt,nk1,nk2)=Ftkk_detrended.shape;
    Ftkk=np.empty((nt,nk1,nk2));
    
    #TODO: use vectorized numpy
    for k1 in range(nk1):
        for k2 in range(nk2):
            Ftkk[:,k1,k2] = recombine_trend_unscaled(
                Ftkk_detrended[:,k1,k2],bkk[k1,k2],Ckk[k1,k2]
            )
    return Ftkk;
    


def kspace_predict_from_trend_unscaled(f0kk, trend,  pred_ts):
    (bkk,Ckk,cycle_length) = trend;

    # Quadratic trend for prediction-times. Note: trend is in original time-scale
    quadratic_trend=bkk[:,:,0,None]+bkk[:,:,1,None]*pred_ts[None,None,:]+bkk[:,:,2,None]*pred_ts[None,None,:]*pred_ts[None,None,:];

    # Get new coefficients for trend, shifting time-scale to t0=0
    (nk1,nk2) = f0kk.shape;
    bkk_new=np.array([[polynomial_trend(quadratic_trend[i,j],2) for j in range(nk2)] for i in range(nk1)])

    
    one = np.ones(len(pred_ts)); # Set Ftkk_detrended to constant f0kk for all t.
    Ftkk_detrended = f0kk[None,:,:]*one[:,None,None];
    Ftkk_retrended = recombine_trends_unscaled(Ftkk_detrended,bkk_new,Ckk)

    return Ftkk_retrended;    
    

def predict_from_trend_unscaled(training_Ftxx,cycle_length,pred_length):
    (nk1,nk2)     = training_Ftxx.shape[1:]; # For now
    training_Ftkk = dctn(training_Ftxx,norm='ortho',axes=[1,2])[:,:nk1,:nk2];

    (Ftkk_detrended,bkk,Ckk) =  separate_trends_unscaled(training_Ftkk,cycle_length);

    t0                = training_Ftxx.shape[0];      # Where we start predicting from
    training_ts       = np.arange(0,t0);    # ts corresponding to training
    pred_ts           = np.arange(t0,t0+pred_length);
    
    trend_descriptors = (bkk,Ckk,cycle_length);
    f0kk              = Ftkk_detrended[-1];  # We start where the training data ends

    predicted_Ftkk = kspace_predict_from_trend_unscaled(f0kk,trend_descriptors, pred_ts);

    predicted_Ftxx = idctn(predicted_Ftkk,norm='ortho',axes=[1,2]);

    return predicted_Ftxx, predicted_Ftkk;

    
