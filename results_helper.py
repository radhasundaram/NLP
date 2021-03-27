import datetime
import logging

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from ta_lib.core import dataset
from ta_lib.core.api import create_context, initialize_environment
from ta_lib.core.utils import get_package_path
from ta_lib.tpo.optimization.discrete.stage_3_tpo_optimizer import (
    Retailer_Price_Fn,
    Retailer_Unit_Sales_Stg3_Fn,
    get_TPR_spend_coeff,
)


def current_time():
    return datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")


def get_pack_path():
    """Absolute path for the package."""
    return get_package_path().replace("\\", "/").replace("src", "")


logger = logging.getLogger(__name__)

initialize_environment(debug=False, hide_warnings=True)
config_path = get_pack_path() + "notebooks/tpo/python/conf/config.yml"  # noqa
context = create_context(config_path)
logger = logging.getLogger(__name__)


# ################################ Event Price Calender Export ######################
def Event_Price_Calendar_Fn(Globals, EDLP, TPR, FLAG, Export=True):
    """To Generate Optimized event calender as a csv.

    Parameters
    ----------
    Globals : Globally defined variables being passed as an argument for the calculation
    EDLP : np.Array / List of Lists containing Optimized EDLP Values
    TPR  : np.Array / List of Lists containing Optimized TPR Values
    FLAG : np.Array / List of Lists containing Optimized FLAG Values
    Export : Boolean, a Flag to mention, whether to export it as external csv file
    Returns:
    --------
    DataFrame containing Optimized Event Price Calendar

    """
    inp = np.zeros((Globals.Tot_Week, Globals.Tot_Prod * 2))
    cols = [
        (
            f"{Globals.Mapping_reverse[i].replace('_Retailer', '')}_EDLP",
            f"{Globals.Mapping_reverse[i].replace('_Retailer', '')}_TPR",
        )
        for i in range(0, Globals.Tot_Prod)
    ]
    cols = [j for i in cols for j in i]
    df = pd.DataFrame(inp, columns=cols)
    df1 = pd.DataFrame({"Week": [i for i in range(1, Globals.Tot_Week + 1)]})
    for pr in range(len(EDLP)):
        for wk in range(len(EDLP[pr])):
            df.iloc[wk, 2 * pr] = (
                EDLP[pr][wk] * FLAG[pr][wk] * Globals.Base_Price_stg2[pr][wk]
            )

            df.iloc[wk, 2 * pr + 1] = TPR[pr][wk] * (1 - FLAG[pr][wk])
    Event_Price_Calendar = pd.concat([df1, df], 1)
    if Export:
        dataset.save_dataset(
            context, Event_Price_Calendar, "raw/Event_Price", log_time=current_time()
        )  # noqa
    return Event_Price_Calendar


# ########################## Event Spend Calendar Export ############################
def Event_Spend_Calendar_Fn(Globals, Model, Export=True):
    """To Generate Optimized Spend calender as a csv.

    Parameters
    ----------
    Globals : Globally defined variables being passed as an argument for the calculation
    Model : Pyomo Model Object created with the required variables as per the Stages
    Export : Boolean, a Flag to mention, whether to export it as external csv file
    Returns:
    --------
    DataFrame containing Optimized Event Spend Calendar

    """
    Spend_coeff = Globals.TE_Val
    Event_Spend = pd.DataFrame({"Week": range(1, Globals.Tot_Week + 1)})
    for Cur_Ret in range(Globals.Tot_Prod):

        PPG_EDLP_Event_Spend = [
            pyo.value(
                Spend_coeff[Cur_Ret][0]
                * Retailer_Unit_Sales_Stg3_Fn(Model, Cur_Ret, Wk)
                * (Globals.FLAG_Inter[Cur_Ret, Wk])
            )
            for Wk in Model.Wk_index
        ]

        PPG_TPR_Event_Spend = [
            pyo.value(
                get_TPR_spend_coeff(Model, Cur_Ret, Wk)
                * Retailer_Unit_Sales_Stg3_Fn(Model, Cur_Ret, Wk)
                * (1 - Globals.FLAG_Inter[Cur_Ret, Wk])
            )
            for Wk in range(Globals.Tot_Week)
        ]

        name = f"{Globals.Mapping_reverse[Cur_Ret].replace('_Retailer', '')}"
        Event_Spend[name + "_EDLP_Spend"] = PPG_EDLP_Event_Spend
        Event_Spend[name + "_TPR_Spend"] = PPG_TPR_Event_Spend
        if Export:
            dataset.save_dataset(
                context, Event_Spend, "raw/Event_Spend", log_time=current_time()
            )
    return Event_Spend


# ############################## Summary Export Function ############################
def Summary_Export_Fn(Globals, Model, Export=True):
    """To Generate Summary as a csv only when the Optimizer is truly converging.

    Parameters
    ----------
    Globals : Globally defined variables being passed as an argument for the calculation
    Model : Pyomo Model Object created with the required variables as per the Stages
    Export : Boolean, a Flag to mention, whether to export it as external csv file
    Returns:
    --------
    DataFrame containing Summary Exporting

    """
    Spend_coeff = Globals.TE_Val
    Summary = pd.DataFrame(
        {
            "Promotion Event": [
                "Optimized TPR Spend",
                "Optimized EDLP Spend",
                "Total Promotional Events",
            ]
        }
    )
    for Cur_Ret in range(Globals.Tot_Prod):
        Opt_EDLP = pyo.value(
            sum(
                [
                    Spend_coeff[Cur_Ret][0]
                    * Retailer_Unit_Sales_Stg3_Fn(Model, Cur_Ret, Wk)
                    * (Globals.FLAG_Inter[Cur_Ret, Wk])
                    for Wk in Model.Wk_index
                ]
            )
        )
        Opt_TPR = pyo.value(
            sum(
                [
                    get_TPR_spend_coeff(Model, Cur_Ret, Wk)
                    * Retailer_Unit_Sales_Stg3_Fn(Model, Cur_Ret, Wk)
                    * (1 - Globals.FLAG_Inter[Cur_Ret, Wk])
                    for Wk in range(Globals.Tot_Week)
                ]
            )
        )
        Promo_Events = pyo.value(
            sum(
                [
                    (1 - Globals.FLAG_Inter[Cur_Ret, Wk])
                    for Wk in range(Globals.Tot_Week)
                ]
            )
        )
        name = f"{Globals.Mapping_reverse[Cur_Ret].replace('_Retailer', '')}"
        Summary[name] = [Opt_TPR, Opt_EDLP, Promo_Events]
        if Export:
            dataset.save_dataset(
                context, Summary, "raw/Summary_Export", log_time=current_time()
            )

    return Summary


# ########################## Baseline Vs Increment Export ###########################
def Base_Increment_Fn(Globals, Model, runtime, Export=True):
    pd.set_option("display.float_format", lambda x: "%.5f" % x)
    baseline = dataset.load_dataset(context, "TPO_Input/Base_Output")
    baseline_array = np.array(
        baseline[
            (baseline.Retailer == "Retailer" + str(Globals.Ret))
            & (baseline.Category == "Category" + str(Globals.Cat))
        ]
    )[0]
    Baseline_Sales = baseline_array[2]
    Optimized_Sales = pyo.value(Model.Obj)
    Incremental_Sales = Optimized_Sales - Baseline_Sales
    Incremental_Perc = str(Incremental_Sales * 100 / Baseline_Sales) + " %"
    Trade_Spend = pyo.value(
        sum(
            [
                sum(
                    [
                        (
                            Globals.Base_Price_stg2[Cur_Ret][Wk]
                            - Retailer_Price_Fn(Model, Cur_Ret, Wk)
                        )
                        * Retailer_Unit_Sales_Stg3_Fn(Model, Cur_Ret, Wk)
                        for Wk in range(Globals.Tot_Week)
                    ]
                )
                for Cur_Ret in range(Globals.Tot_Prod)
            ]
        )
    )
    Incremental_Sales_Rep = pd.DataFrame(
        {
            "Baseline_Sales": [Baseline_Sales],
            "Optimized_Sales": [Optimized_Sales],
            "Trade_Spend": [Trade_Spend],
            "Incremental_Sales": [Incremental_Sales],
            "Incremental_Sales_%": [Incremental_Perc],
            "Retailer": Globals.Ret,
            "Category": Globals.Cat,
            "Runtime": runtime,
        }
    )

    if Export:
        dataset.save_dataset(
            context,
            Incremental_Sales_Rep,
            "raw/Base_Increment",
            log_time=current_time(),
        )

    return Incremental_Sales_Rep


def Dollar_Sales_Export_Fn(Globals, Model, Export=True):
    """To Generate Dollar_Sales as a csv only when the Optimizer is truly converging.

    Parameters
    ----------
    Globals : Globally defined variables being passed as an argument for the calculation
    Model : Pyomo Model Object created with the required variables as per the Stages
    Export : Boolean, a Flag to mention, whether to export it as external csv file
    Returns:
    --------
    DataFrame containing Dollar Sales Exporting

    """
    Doll_Sales = [
        pyo.value(Retailer_Unit_Sales_Stg3_Fn(Model, P, W))
        * pyo.value(Retailer_Price_Fn(Model, P, W))
        for P in Model.PPG_index
        for W in Model.Wk_index
    ]

    Doll_Sales_DF = pd.DataFrame({"Dollar_Sales": Doll_Sales})

    Doll_Sales_DF["Product"] = [
        f"Product{i}" for i in Model.PPG_index for j in Model.Wk_index
    ]

    Doll_Sales_DF["Week"] = list((range(1, Globals.Tot_Week + 1))) * Globals.Tot_Prod

    if Export:
        dataset.save_dataset(
            context, Doll_Sales_DF, "raw/Dollar_Sales", log_time=current_time()
        )

    return Doll_Sales_DF


def save_results(Globals, runtime):
    """Save all the Results.

    Parameters
    ----------
    Globals : Globally defined variables being passed as an argument for the calculation
    runtime : Boolean, True to run the Base_Increment_Fn function and False to not

    Helper function to generate a 4 fold report of the optimizer outputs

    """
    try:
        Event_Price_Calendar_Fn(
            Globals, Globals.EDLP_Final, Globals.TPR_Final, Globals.FLAG_Inter
        )
    except Exception as e:
        logger.error("\n Error in saving Price results : " + str(e))

    try:
        Event_Spend_Calendar_Fn(Globals, Globals.Solver_Model_Result)

    except Exception as e:
        logger.error("\n Error in saving Spend results : " + str(e))

    try:
        Summary_Export_Fn(Globals, Globals.Solver_Model_Result)

    except Exception as e:
        logger.error("\n Error in saving Summary results : " + str(e))

    try:
        Dollar_Sales_Export_Fn(Globals, Globals.Solver_Model_Result)

    except Exception as e:
        logger.error("\n Error in saving Dollar Sales results : " + str(e))

    try:
        Base_Increment_Fn(Globals, Globals.Solver_Model_Result, runtime=runtime)

    except Exception as e:
        logger.error("\n Error in saving Base Incremental results : " + str(e))
