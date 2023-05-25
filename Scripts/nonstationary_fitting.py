
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt

from scipy.stats import norm, gamma, lognorm, genextreme, genextreme as gev
from scipy.optimize import minimize
from xclim.indices.stats import fit

import random

# import sys; sys.path.append('/home/clair/wwa'); from wwa import xyline

#######################################################################################################################################
## FITTING METHODS

def ns_mle(x0, covariate, x, dist, fittype):
    
    # Generic fitting method: can add extra distributions & fit types as needed
    
    # unpack nonstationary parameters
    if dist in [norm, lognorm]:
        mu, sigma, alpha = x0[:3]
    elif dist in [gev, genextreme, gamma]:
        mu, sigma, alpha, shape = x0[:4]
            
    # convert to vector of stationary loc & scale
    if fittype == "shift":
        loc = mu + alpha * covariate
        scale = sigma
    elif fittype == "fixeddisp":
        loc = mu * np.exp(alpha * covariate / mu)
        scale = sigma * np.exp(alpha * covariate / mu)
    elif fittype == "scale":
        loc = mu
        scale = sigma + alpha * covariate
    elif fittype == "shiftscale":
        loc = mu + alpha * covariate
        scale = sigma + x0[-1] * covariate
    else:
        print("Fit type not known: choose from shift, fixeddisp, scale, shiftscale")
        return
        
    # pack stationary parameters  
    if dist == lognorm:
        pars = [scale, 0, np.exp(loc)]      # python uses an odd parametrisation, this gives same results as R
    elif dist in [gev, genextreme, gamma]:
        pars = [shape, loc, scale]
    else:
        pars = [loc, scale]
    
    # return negative log-likelihood
    return -dist.logpdf(x, *pars).sum()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Univariate fitting method for Pandas DataFrame
def ns_fit(dist, fittype, data, cov_name, var_name, solver = "Nelder-Mead", **optim_kwargs):
    
    data = data.dropna(axis = 0, how = "any")
    
    # initial parameters need to be passed as mu, sigma, alpha, shape, beta
    covariate = data[cov_name]
    x = data[var_name]
    
    # currently no option to provide initial estimates - use stationary parameters as initial fit
    if dist == norm: 
        init = [x.mean(), x.std(), 0]
    elif dist == lognorm:
        init = [np.log(x).mean(), np.log(x).std(), 0]
    elif dist in [gev, genextreme, gamma]:
        shape, loc, scale = dist.fit(x)
        init = [loc, scale, 0, shape]
    else:
        print(dist.name+" distribution not yet implemented")
        return
    
    # if needed, add beta to initial parameters
    if fittype == "shiftscale": init = init + [0]
    
    ml_fit = minimize(ns_mle, init, args = (covariate, x, dist, fittype), method = solver, **optim_kwargs)
    
    # add named parameters to output, to avoid any possible confusion
    parnames = ["mu", "sigma", "alpha", "shape", "beta"][:len(ml_fit.x)]
    ml_fit["pars"] = {parnames[i] : ml_fit.x[i] for i in range(len(parnames))}
    
    res = {"results" : ml_fit, "dist" : dist.name, "fittype" : fittype, "cov_name" : cov_name, "var_name" : var_name, "data" : data,
           "solver" : solver, "kwargs" : optim_kwargs}
    
    return res


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Wrapper for ufunc

def ns_cellfit(x, covariate, init, dist, fittype, solver = "Nelder-Mead", **optim_kwargs):
            
    # additional processing step needed to handle missing data & extract output per cell
    xx = x[~np.isnan(x)]
    if len(xx) < len(x)/2: 
        return np.array([np.nan]*(len(init)+2))
    else:
        fitted = minimize(ns_mle, init, args = (covariate, x, dist, fittype), method = solver, **optim_kwargs)
        # store convergence indicator, nll, fitted parameters
        return np.array([fitted["status"], fitted["fun"]] + list(fitted["x"]))
    
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ufunc method to fit DataArray

def ns_mapfit(dist, fittype, da, covariate, solver = "Nelder-Mead", **optim_kwargs):
        
    zeros = xr.zeros_like(da.isel(time = 0)).squeeze(drop = True)
    
    # currently no option to provide initial estimates as external argument - use stationary parameters as initial fit
    if dist == norm: 
        init = [da.mean("time"), da.std("time"), zeros]
    elif dist == lognorm:
        init = [np.log(da).mean("time"), np.log(da).std("time"), zeros]
    elif dist in [gamma]:
        shape, loc, scale = fit(da, dist = "gamma", method = "PWM")
        shape, loc, scale = [par.reset_coords(drop = True) for par in [shape, loc, scale]]
        init = [loc, scale, zeros, shape]
    elif dist in [gev, genextreme]:
        shape, loc, scale = fit(da, dist = "genextreme", method = "PWM")
        shape, loc, scale = [par.reset_coords(drop = True) for par in [shape, loc, scale]]
        init = [loc, scale, zeros, shape]
    else:
        print(dist.name+" distribution not yet implemented")
        return
    
    # if needed, add beta to initial parameters
    if fittype == "shiftscale": init = init + [zeros]
    
    # concatenate initial params into single DataArray
    init = xr.concat(init, "params")
    
    # now use ufunc + wrapper function to perform optimisation step
    ml_fit = xr.apply_ufunc(lambda x, init : ns_cellfit(x, covariate = covariate.values, init = init, dist = dist, fittype = fittype, solver = solver, **optim_kwargs),
                            da, init,
                            input_core_dims=[["time"],["params"]], output_core_dims = [["pars"]], vectorize = True)
    
    # add named parameters to output, to avoid any possible confusion
    ml_fit = ml_fit.assign_coords(pars = ["status", "nll", "mu", "sigma", "alpha", "shape", "beta"][:len(ml_fit.pars)]).rename("ml_fit")
    ml_fit = ml_fit.assign_attrs(dist = dist.name, fittype = fittype, solver = solver)
    
    if len(optim_kwargs) > 0:
        print("Need to assign optim_kwargs as attributes - not yet implemented")
        
    return xr.merge([ml_fit, covariate.rename("covariate")])


#######################################################################################################################################
## SUPPORT METHODS (UNIVARIATE)

def pack_pars(pars, dist):
    
    # pack stationary parameters to pass to distribution: order depends on distribution used
    if dist == lognorm:
        pars = {"shape" : pars["scale"], "loc" : 0, "scale" : np.exp(pars["loc"])}
    elif dist in [gev, genextreme, gamma]:
        pars = {k : pars[k] for k in ["shape", "loc", "scale"]}
    else:
        pars = {k : pars[k] for k in ["loc", "scale"]}
        
    return pars



def ns_pars(mdl, covariate = None, packed = False):
    
    # method to convert parameters to nonstationary location, scale etc
    
    if type(mdl) == xr.core.dataset.Dataset: mdl = mdl.ml_fit
    
    pars = mdl["results"].pars
    
    # if no covariate provided, return result for all values used in fitting
    if not covariate: covariate = mdl["data"][mdl["cov_name"]]
    
    if mdl["fittype"] == "shift":
        loc = pars["mu"] + pars["alpha"] * covariate
        scale = pars["sigma"]
    elif mdl["fittype"] == "fixeddisp":
        loc = pars["mu"] * np.exp(pars["alpha"] * covariate / pars["mu"])
        scale = pars["sigma"] * np.exp(pars["alpha"] * covariate / pars["mu"])
    elif mdl["fittype"] == "scale":
        loc = pars["mu"]
        scale = pars["sigma"] + pars["alpha"] * covariate
    elif mdl["fittype"] == "shiftscale":
        loc = pars["mu"] + pars["alpha"] * covariate
        scale = pars["sigma"] + pars["beta"] * covariate
        
    if "shape" in pars.keys():
        pars = { "loc" : loc, "scale" : scale, "shape" : pars["shape"] }
    else:
        pars = { "loc" : loc, "scale" : scale }
        
    if packed:       
        return pack_pars(pars, eval(mdl["dist"]))
    else:
        return pars
        
    


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def return_level(mdl, rp, covariate = None, lower = False):
    
    pars = ns_pars(mdl, covariate = covariate, packed = True).values()
    dist = eval(mdl["dist"])        
    
    # get return value for return period (scipy doesn't have argument to look at either tail)
    if lower:
        rl = dist.ppf(1/rp, *pars)
    else:
        rl = dist.isf(1/rp, *pars)
    
    return rl


def delta_I(mdl, rp, cov1, cov2, lower = False, relative = False):
    
    rl1 = return_level(mdl, rp, cov1, lower = lower)
    rl2 = return_level(mdl, rp, cov2, lower = lower)
    
    if relative:
        return (rl1 - rl2) / rl2 * 100
    else:
        return rl1 - rl2
    
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def return_period(mdl, event_value, covariate, lower = False):
        
    pars = ns_pars(mdl, covariate = covariate, packed = True).values()
    dist = eval(mdl["dist"])
        
    # get exceedance probability for given event (scipy doesn't have argument to look at either tail)
    if lower:
        ep = dist.cdf(event_value, *pars)
    else:
        ep = dist.sf(event_value, *pars)
        
    # get return period
    return 1/ep


def prob_ratio(mdl, event_value, cov1, cov2, lower = False):
    
    rp1 = return_period(mdl, event_value, cov1, lower = lower)
    rp2 = return_period(mdl, event_value, cov2, lower = lower)
    
    return(rp2/rp1)


#######################################################################################################################################
## REPORTING METHODS

def model_results(mdl, cov1, cov2, event_value = np.nan, rp = np.nan, lower = False, relative_deltaI = False):
    
    # extract numerical results for a single model fit
            
    params = mdl["results"]["pars"]
    
    nspars_1 = {k+"1" : v for k, v in ns_pars(mdl, cov1).items()}
    nspars_1["disp1"] = nspars_1["scale1"] / nspars_1["loc1"]
    nspars_2 = {k+"2" : v for k, v in ns_pars(mdl, cov2).items()}
    nspars_2["disp2"] = nspars_2["scale2"] / nspars_2["loc2"]

    rp1, rp2 = [return_period(mdl, event_value, cov, lower = lower) for cov in [cov1, cov2]]
    pr = rp2/rp1
    rl1, rl2 = [return_level(mdl, rp, cov, lower = lower) for cov in [cov1, cov2]]
    dI = delta_I(mdl, rp, cov1, cov2, lower = lower, relative = relative_deltaI)
    
    res = { "fixed_value" : event_value, "rp_1" : rp1, "rp_2" : rp2, "pr" : pr,
            "fixed_rp" : rp, "rl_1" : rl1, "rl_2" : rl2, "dI" : dI }

    return params | nspars_1 | nspars_2 | res


def boot_results(mdl, cov1, cov2, event_value = np.nan, rp = np.nan, lower = False, relative_deltaI = False, nsamp = 1000, seed = 1):
    
    # General method to bootstrap all numerical results (contents determined by model_results method)
    
    random.seed(seed)
    
    est = pd.DataFrame(model_results(mdl, cov1, cov2, event_value, rp, lower, relative_deltaI), index = ["est"])
    nr = len(mdl["data"])
    
    boot_res = []
    for i in range(nsamp):
        
        # resample the data with replacement, compute the results & add to dataframe
        boot_df = mdl["data"].iloc[random.choices(range(nr), k = nr),:]
        boot_fit = ns_fit(eval(mdl["dist"]), mdl["fittype"], boot_df, mdl["cov_name"], mdl["var_name"], solver = mdl["solver"], **mdl["kwargs"])
        boot_res.append(model_results(boot_fit, cov1, cov2, event_value, rp, lower, relative_deltaI))
        
    boot_res = pd.DataFrame(boot_res).quantile([0.025, 0.975])
    
    return pd.concat([est, boot_res])



#######################################################################################################################################
## PLOTTING METHODS

def trendplot(mdl, cov1, cov2, loc1 = None, loc2 = None, lower = False, ax = None, legend = True):
    
    # extract necessary info from model
    covariate = mdl["data"][[mdl["cov_name"]]].values.flatten()
    x = mdl["data"][[mdl["var_name"]]].values.flatten()
    loc = ns_pars(mdl)["loc"]
    event_value = x[covariate == cov1]
    
    # if bounds for location not provided, could also use small bootstrap sample to compute
    if loc1 is None: loc1 = [ns_pars(mdl, cov1)["loc"]] + [np.nan]*2
    if loc2 is None: loc2 = [ns_pars(mdl, cov2)["loc"]] + [np.nan]*2
        
    # if log distribution, convert location to native units
    if mdl["dist"] in ["lognorm"]: loc, loc1, loc2 = [np.exp(l) for l in [loc, loc1, loc2]]
    
    # and now, plotting
    if not ax: fig, ax = plt.subplots(figsize = (5,3))
    
    # observed points
    ax.scatter(covariate, x, color = "k", marker = ".")
    ax.scatter(cov1, event_value, color = "magenta", marker = "o")
    
    # fitted location and return levels
    # xyline(covariate, loc, ax, 2, ls = "--", color = "k", label = "$\mu'$")
    # xyline(covariate, return_level(mdl, 6, lower = lower), ax, ls = "--", color = "blue", label = "6-year event")
    # xyline(covariate, return_level(mdl, 40, lower = lower), ax, ls = "--", color = "blue", alpha = 0.5, label = "40-year event")
    
    # bounds for location
    ax.plot([cov1]*3, loc1, color = "k", marker = "_", ms = 10)
    ax.plot([cov2]*3, loc2, color = "k", marker = "_", ms = 10)
    
    ax.set_xlabel("GMST anomaly (smoothed)")
    
    

def rlplot(mdl, cov1, cov2, event_value, lower = False, ax = None, ci_nsamp = 10, legend = True, seed = 1):
    
    random.seed(seed)
    
    # define values at which quantities are to be evaluated & plotted
    x_obs = 1/np.linspace(1,0,num = len(mdl["data"])+1, endpoint = False)[1:]
    x_est = np.array(list(np.arange(1.1,2,0.1)) + list(range(2,100)) + list(range(100,1000,10)) + list(range(1000,10000,100)))
    x_ci = np.array([10,20,50,100,200,500,1000,2000,5000,10000])
    
    # # if log distribution, convert location to native units
    # if mdl["dist"] in ["lognorm"]: loc, loc1, loc2 = [np.exp(l) for l in [loc, loc1, loc2]]
    
    # and now, plotting
    if not ax: fig, ax = plt.subplots(figsize = (5,3), dpi = 100)
    
    ax.semilogx(x_obs, sorted(stransf(mdl, cov1, lower = lower), reverse = lower), ls = "", marker = ".", color = "firebrick")
    ax.plot(x_obs, sorted(stransf(mdl, cov2, lower = lower), reverse = lower), ls = "", marker = ".", color = "blue")

    ax.axhline(event_value, color = "magenta", label = "Observed event", lw = 1)
    ax.plot(x_est, return_level(mdl, x_est, cov1, lower = lower), color = "firebrick", label = "Event in current climate")
    ax.plot(x_est, return_level(mdl, x_est, cov2, lower = lower), color = "blue", label = "Counterfactual event")
    
    # if not provided, use small bootstrap sample to get confidence intervals
    df = mdl["data"][[mdl["cov_name"], mdl["var_name"]]]
    boot_ci_1 = []
    boot_ci_2 = []
    for i in range(ci_nsamp):
        boot_df = df.iloc[np.random.choice(len(df), size = len(df)),:]
        boot_fit = ns_fit(eval(mdl["dist"]), mdl["fittype"], boot_df, mdl["cov_name"], mdl["var_name"], solver = mdl["solver"], **mdl["kwargs"])
        boot_ci_1.append(return_level(boot_fit, x_ci, covariate = cov1, lower = lower))
        boot_ci_2.append(return_level(boot_fit, x_ci, covariate = cov2, lower = lower))

    ci1 = np.quantile(np.column_stack(boot_ci_1), [0.025, 0.975], axis = 1)
    ci2 = np.quantile(np.column_stack(boot_ci_2), [0.025, 0.975], axis = 1)
    
    ax.plot(x_ci, ci1.transpose(), color = "firebrick", ls = "--", alpha = 0.5)
    ax.plot(x_ci, ci2.transpose(), color = "blue", ls = "--", alpha = 0.5)
    
    # add rug
    rp1, rp2 = [return_period(mdl, covariate = c, event_value = event_value, lower = lower) for c in [cov1, cov2]]
    
    y0 = ax.get_xlim()[0]
    # ax.plot(rp1, y0, marker = "|", mew = 3, ms = 10, color = "firebrick")
    # ax.plot(rp2, y0, marker = "|", mew = 3, ms = 10, color = "blue")
    
    ax.set_xlim(None, 10e3)
    ax.set_xlabel("Return period (years)")
    
    if legend:
        ax.legend()
    
    
def return_ci(mdl, cov1, lower = False, ci_nsamp = 10, seed = 1):
    
    random.seed(seed)
    
    # define values at which quantities are to be evaluated & plotted
    x_ci = np.array([10,20,50,100,200,500,1000,2000,5000,10000])
    
    # if not provided, use small bootstrap sample to get confidence intervals
    df = mdl["data"][[mdl["cov_name"], mdl["var_name"]]]
    boot_ci_1 = []
    for i in range(ci_nsamp):
        boot_df = df.iloc[np.random.choice(len(df), size = len(df)),:]
        boot_fit = ns_fit(eval(mdl["dist"]), mdl["fittype"], boot_df, mdl["cov_name"], mdl["var_name"], solver = mdl["solver"], **mdl["kwargs"])
        boot_ci_1.append(return_level(boot_fit, x_ci, covariate = cov1, lower = lower))

    ci1 = np.quantile(np.column_stack(boot_ci_1), [0.025, 0.975], axis = 1)
    return ci1
    
    # ax.plot(x_ci, ci1.transpose(), color = "firebrick", ls = "--", alpha = 0.5)
    # ax.plot(x_ci, ci2.transpose(), color = "blue", ls = "--", alpha = 0.5)
    

    
    
    
#######################################################################################################################################
## SUPPORT METHODS (MAPS)

def ns_parmap(mdl, covariate = None):
        
    if mdl.fittype == "shift":
        loc = mdl.sel(pars = "mu") + mdl.sel(pars = "alpha") * covariate
        scale = mdl.sel(pars = "sigma").reset_coords(drop = True)
    elif mdl.fittype == "fixeddisp":
        loc = mdl.sel(pars = "mu") * np.exp(mdl.sel(pars = "alpha") * covariate / mdl.sel(pars = "mu"))
        scale = mdl.sel(pars = "sigma") * np.exp(mdl.sel(pars = "alpha") * covariate / mdl.sel(pars = "mu"))
    elif mdl.fittype == "scale":
        loc = mdl.sel(pars = "mu").reset_coords(drop = True)
        scale = mdl.sel(pars = "sigma") + mdl.sel(pars = "alpha") * covariate
    elif mdl.fittype == "shiftscale":
        loc = mdl.sel(pars = "mu") + mdl.sel(pars = "alpha") * covariate
        scale = mdl.sel(pars = "sigma") + mdl.sel(pars = "beta") * covariate
        
    loc = loc.rename("location").assign_attrs(long_name = "Location parameter").squeeze(drop = True).reset_coords(drop = True)
    scale = scale.rename("scale").assign_attrs(long_name = "Scale parameter").squeeze(drop = True).reset_coords(drop = True)
    
    if "shape" in mdl.pars:
        return xr.Dataset({ "location" : loc, "scale" : scale, "shape" : mdl.sel(pars = "shape", drop = True).reset_coords(drop = True).rename("shape").assign_attrs(long_name = "Shape parameter", units = "") })
    else:
        return xr.Dataset({ "location" : loc, "scale" : scale })
    

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
def rlmap(mdl, rp, covariate, lower = False):
    
    dist = eval(mdl.dist)
    pars = ns_parmap(mdl, covariate).squeeze(drop = True)
    
    if type(rp) in [int, float]: rp = xr.ones_like(mdl.isel(pars = 0)) * rp
    rp = rp.squeeze(drop = True)
        
    # pack stationary parameters: order depends on distribution used
    if dist == lognorm:
        pars = [pars["scale"], xr.zeros_like(pars["scale"]), np.exp(pars["location"])]
    elif dist in [gev, genextreme, gamma]:
        pars = [pars["shape"], pars["location"], pars["scale"]]
    else:
        pars = [pars["location"], pars["scale"]]
        
    if lower:
        rl = xr.apply_ufunc(lambda rp, pars : dist.ppf(1/rp, *pars), rp, xr.concat(pars, "params"), input_core_dims = [[], ["params"]], vectorize = True)
    else:
        rl = xr.apply_ufunc(lambda rp, pars : dist.isf(1/rp, *pars), rp, xr.concat(pars, "params"), input_core_dims = [[], ["params"]], vectorize = True)
        
    return xr.DataArray(rl, dims = mdl.isel(pars = 0).dims, coords = mdl.isel(pars = 0).coords).reset_coords(drop = True).assign_attrs(long_name = "Return level", units = mdl.units).rename("rl")


def dImap(mdl, rp, cov1, cov2, lower = False, relative = False):
    
    
    rl1 = rlmap(mdl, rp, cov1, lower = lower)
    rl2 = rlmap(mdl, rp, cov2, lower = lower)
    
    if relative:
        ((rl1 - rl2) / rl2 * 100).rename("delta_I_%").assign_attrs(long_name = "Relative change in intensity", units = "%")
    else:
        return (rl1 - rl2).rename("delta_I").assign_attrs(long_name = "Change in intensity")
    

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def rpmap(mdl, event_value, covariate, lower = False):
    
    dist = eval(mdl.dist)
    pars = ns_parmap(mdl, covariate).squeeze(drop = True)
    
    if type(event_value) in [int, float]: event_value = xr.ones_like(mdl.isel(pars = 0)) * event_value
    event_value = event_value.squeeze(drop = True)
            
    # pack stationary parameters: order depends on distribution used
    if dist == lognorm:
        pars = [pars["scale"], xr.zeros_like(pars["scale"]), np.exp(pars["location"])]
    elif dist in [gev, genextreme, gamma]:
        pars = [pars["shape"], pars["location"], pars["scale"]]
    else:
        pars = [pars["location"], pars["scale"]]
        
    # get exceedance probability for given event (scipy doesn't have argument to look at either tail)
    if lower:
        ep = xr.apply_ufunc(lambda ev, pars : dist.cdf(ev, *pars), event_value, xr.concat(pars, "params"), input_core_dims = [[],["params"]], vectorize = True)
    else:
        ep = xr.apply_ufunc(lambda ev, pars : dist.sf(ev, *pars), event_value, xr.concat(pars, "params"), input_core_dims = [[],["params"]], vectorize = True)
    
    return 1/ep.rename("rp").assign_attrs(long_name = "Return period", units = "years").reset_coords(drop = True)


def prmap(mdl, event_value, cov1, cov2, lower = False):
    
    rp1 = rpmap(mdl, event_value, cov1)
    rp2 = rpmap(mdl, event_value, cov2)

    return (rp2 / rp1).rename("prob_ratio").assign_attrs(long_name = "Probability ratio", units = "")


#######################################################################################################################################
## TRANSFORM DATA TO STATIONARITY

def stransf(mdl, covariate = None, lower = False):
    
    # use PIT to transform to standard distribution

    pars = ns_pars(mdl, packed = True)
    x = mdl["data"][[mdl["var_name"]]].values.flatten()
    dist = eval(mdl["dist"])
    
    # parameters of target stationary distribution (set to standard form if not provided)
    if not covariate:
        s_pars = {"lognorm" : [1], "gev" : [0], "genextreme" : [0], "gamma" : [1], "norm" : []}[dist.name]
    else:
        s_pars = ns_pars(mdl, covariate, packed = True).values()

    # pack stationary parameters to pass to distribution: order depends on distribution used
    if dist == lognorm:
        pars = [pars["scale"], 0, np.exp(pars["loc"])]
    elif dist in [gev, genextreme]:
        pars = [pars["shape"], pars["loc"], pars["scale"]]
    elif dist in [gamma]:
        pars = [pars["shape"], pars["loc"], pars["scale"]]
    else:
        pars = [pars["loc"], pars["scale"]]
        
    # get PIT for given event (scipy doesn't have argument to look at either tail)
    if lower:
        ep = dist.cdf(x, *pars)
        pit = dist.ppf(ep, *s_pars)
    else:
        ep = dist.sf(x, *pars)
        pit = dist.isf(ep, *s_pars)
        
    return pit