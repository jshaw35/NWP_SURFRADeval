"""
Test statistical significance of differences between RRFS and HRRR forecasts. Consider adding RRFS ens. 2 since it is closest to the ultimately chosen RRFS control members.
"""

# %%
import pandas as pd
import scipy.stats as stats
import xarray as xr

# %%
if __name__ == "__main__":
    error_stddevs_conditions = pd.read_csv("data/figure_outputs/error_stddev_cloud_conditions.csv", index_col=[0, 1]).to_xarray()
    N_stddevs_conditions = pd.read_csv("data/figure_outputs/Nhours_cloud_conditions.csv", index_col=[0]).to_xarray()

    # %%
    # Perform an f-test to compare the variances of two groups.
    # Use an f-test to compare the variances of two groups
    # Null hypothesis: The variances of the two groups are equal
    pvalue_list = []
    for _var in error_stddevs_conditions.data_vars:
        for _nwp_source in error_stddevs_conditions.nwp_source:
            error_stddev = error_stddevs_conditions[_var].drop_sel(nwp_source=_nwp_source.values)
            df = N_stddevs_conditions[_var] - 1

            var = error_stddev**2 # Variance is the square of the standard deviation
            # F must be the ratio of the higher to the lower variance (doesn't make sense why)
            var1 = var.max(dim='nwp_source')
            var2 = var.min(dim='nwp_source')

            F = var1 / var2
            # p_value = 1 - stats.f.cdf(F, df, df) 
            p_value = stats.f.sf(F, df, df)
            # The CDF is the likelihood that a value is less than or equal to the given value, so we subtract from 1 to get the p-value for the right tail of the distribution. The p_value is the probability of observing a value as or more extreme than F under the null hypothesis of equal variances. When the p_value is small, it indicates that the observed F is unlikely under the null hypothesis, suggesting that the variances are significantly different.
            p_value_da = xr.DataArray(
                p_value,
                coords={
                    "location": error_stddevs_conditions.location,
                },
                dims=["location"],
            )
            p_value_da = p_value_da.assign_coords(
                {
                    "nwp_source": _nwp_source.values,
                    "variable": _var,
                }
            ).expand_dims(["nwp_source", "variable"])
            pvalue_list.append(p_value_da)
    p_value_ds = xr.combine_by_coords(pvalue_list)
    # %%
    # Re-label the nwp_source dimension to indicate the models compared rather than the model that is excluded.
    relabels = {}
    sources = list(error_stddevs_conditions.nwp_source.values)
    for _nwp_source in error_stddevs_conditions.nwp_source:
        true_sources = sources.copy()
        true_sources.remove(_nwp_source.values)
        new_label = f'{true_sources[0]} vs. {true_sources[1]}'
        relabels[str(_nwp_source.values)] = new_label
    p_value_ds["nwp_source"] = list(relabels.values())
    p_value_ds.name = "p_value"
    p_value_ds.to_dataframe().to_csv("data/figure_outputs/p_values_f_test_stddevs.csv")
# %%
