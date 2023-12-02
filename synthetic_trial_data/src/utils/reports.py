import pandas as pd
from typing import List
from ydata_profiling import ProfileReport

def visual_time_series_report(
    df: pd.DataFrame,
    unique_id: str,
    timeseries_cols: List[str],
    sort_by: str,
    to_notebook_iframe: bool = True
):
    """
    Generate a time-series profile report for a specific patient using 
    pandas-profiling's ProfileReport.

    :param df: The input DataFrame containing patient data.
    :type df: pd.DataFrame
    :param unique_id: Unique identifier for the patient for whom the report 
        is to be generated.
    :type unique_id: str
    :param timeseries_cols: List of column names in the DataFrame that 
        represent time-series data.
    :type timeseries_cols: List[str]
    :param sort_by: Column name used for sorting the data chronologically.
    :type sort_by: str
    :param to_notebook_iframe: If True, the report will be displayed directly in a
        Jupyter notebook using an iframe. Default is True.
    :type to_notebook_iframe: bool
    :return: ProfileReport object containing the time-series report of the 
        specified patient.
    :rtype: ProfileReport

    Example:
    .. code-block:: python

        df = pd.DataFrame({
            'subjid': ['A', 'A', 'B', 'B'],
            'timestamp': ['2022-01-01', '2022-01-02', '2022-01-01', '2022-01-02'],
            'vital1': [98, 99, 95, 96],
            'vital2': [120, 122, 119, 118]
        })

        report = visual_time_series_report(
            df, unique_id='A', timeseries_cols=['vital1', 'vital2'], sort_by='timestamp'
        )

    """

    # Filter for single patient
    time_df = df[df["subjid"] == unique_id]
    
    # Setting what variables are time series
    type_schema = {col_name: "timeseries" for col_name in timeseries_cols}
    
    #Enable tsmode to True to automatically identify time-series variables
    #Provide the column name that provides the chronological order of your time-series
    profile = ProfileReport(
        time_df,
        tsmode=True,
        type_schema=type_schema,
        sortby=sort_by,
        title=f"Time-Series of patient {unique_id}"
    )

    if to_notebook_iframe:
        profile.to_notebook_iframe()

    return profile