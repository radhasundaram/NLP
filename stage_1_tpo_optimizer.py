#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 02/06/20 11:46 PM
# @author: Gurusankar G

from __future__ import division

import logging
import time
from functools import reduce

import pyomo.environ as pyo

from ta_lib.core.utils import get_package_path

logger = logging.getLogger(__name__)


def get_pack_path():
    """Absolute path for the package."""
    return get_package_path().replace("\\", "/").replace("src", "")


def TPR_Conv(Model, P):
    """To pick relevent cutoff value for the chosen TPR value."""
    if Globals.TPR_Cutoff.iloc[P, 0] > 0:

        return Globals.TPR_Cutoff.iloc[P, 0]

    elif pyo.value(Model.TPR[P]) > Globals.TPR_Cutoff.iloc[P, 3]:

        return Globals.TPR_Cutoff.iloc[P, 2]
    else:

        return Globals.TPR_Cutoff.iloc[P, 1]


def Dollar_Sales_Stg1_Fn(Model):
    """Define pyomo callback for calculating dollar sales.

    Parameters
    ----------
    Model : Pyomo Model Object created with the required variables
    Returns:
    --------
    Pyomo Expression, containing Dollar sales equation

    """
    return sum(
        [
            Model.Weeks[P]
            * Retailer_Unit_Sales_Fn_Const1(Model, P)
            * Retailer_Price_Fn_Const1(Model, P)
            + (Globals.Tot_Week - Model.Weeks[P])
            * Retailer_Unit_Sales_Fn_Const2(Model, P)
            * Retailer_Price_Fn_Const2(Model, P)
            for P in Model.PPG_index
        ]
    )


def Retailer_Unit_Sales_Fn_Const1(Model, P):
    """Define pyomo callback for calculating product unit sales.

    Parameters
    ----------
    Model : Pyomo Model Object created with the required variables
    P :    Integer, PPG number to calculate EDLP Unit sales for all
    the weeks. Will be iteratively called by Model Objective function
    Returns:
    --------
    Pyomo Expression, containing product unit sales equation

    """
    Self = pyo.exp(
        pyo.log(Model.EDLP[P] * Globals.Base_Price_stg1[P]) * Globals.EDLP_Coef[P][P]
        + Globals.Intercepts_stg1[P]
    )
    Comp = Competitor_Unit_Effect_Stg1_Fn(Model, P)
    Unit_Sales = Self * Comp
    return Unit_Sales


def Retailer_Unit_Sales_Fn_Const2(Model, P):
    """Define function for calculating product unit sales.

    Parameters
    ----------
    Model : Pyomo Model Object created with the required variables
    P :    Integer, PPG number to calculate TPR Unit sales
    for all the weeks. Will be iteratively called by Model
    Objective function
    Returns:
    --------
    Pyomo Expression, containing product unit sales equation

    """
    Self = pyo.exp(
        pyo.log(Globals.Base_Price_stg1[P]) * Globals.EDLP_Coef[P][P]
        + Model.TPR[P] * TPR_Conv(Model, P)
        + Globals.Intercepts_stg1[P]
    )
    Comp = Competitor_Unit_Effect_Stg1_Fn(Model, P)
    Unit_Sales = Self * Comp
    return Unit_Sales


def Retailer_Price_Fn_Const1(Model, P):
    """Define pyomo callback for calculating product unit price.

    Parameters
    ----------
    Model : Pyomo Model Object created with the required variables
    P :    Integer, PPG number to calculate EDLP Retailer Price for all the weeks.
    Will be iteratively called by Model Objective function
    Returns:
    --------
    Pyomo Expression, containing product Retailer Price equation

    """
    Price = Model.EDLP[P] * Globals.Base_Price_stg1[P]
    return Price


def Retailer_Price_Fn_Const2(Model, P):
    """Define function for calculating product unit price.

    Parameters
    ----------
    Model : Pyomo Model Object created with the required variables
    P :    Integer, PPG number to calculate TPR Retailer Price for all the weeks.
    Will be iteratively called by Model Objective function
    Returns:
    --------
    Pyomo Expression, containing product Retailer Price equation

    """
    Price = Globals.Base_Price_stg1[P] * (1 - Model.TPR[P] / 100)
    return Price


def Competitor_Unit_Effect_Stg1_Fn(Model, Cur_Ret):
    """Define pyomo callback for calculating competitor effect.

    Parameters
    ----------
    Model : Pyomo Model Object created with the required variables
    Cur_Ret :    Integer, PPG number to calculate EDLP Competitor Effect for all the weeks.
    Will be iteratively called by Model Objective function
    Returns:
    --------
    Pyomo Expression, containing Competitor Unit Effect equation

    """

    return 1


# ########################### Trade Spend Constraints #############################
def Total_Trade_Spent_Bnd_Stg1_Fn(Model, Cur_Ret):
    """Define pyomo callback for calculating bound for total trade spent.

    Parameters
    ----------
    Model : Pyomo Model Object created with the required variables
    Cur_Ret :    Integer, PPG number to calculate Total Trade Spent for all the weeks.
    Will be iteratively called by Model Objective function
    Returns:
    --------
    Pyomo Expression, containing product Total Trade Spent equation

    """
    Val = Model.Weeks[Cur_Ret] * (
        Globals.Base_Price_stg1[Cur_Ret] - Retailer_Price_Fn_Const1(Model, Cur_Ret)
    ) * Retailer_Unit_Sales_Fn_Const1(Model, Cur_Ret) + (
        Globals.Tot_Week - Model.Weeks[Cur_Ret]
    ) * (
        Globals.Base_Price_stg1[Cur_Ret] - Retailer_Price_Fn_Const2(Model, Cur_Ret)
    ) * Retailer_Unit_Sales_Fn_Const2(
        Model, Cur_Ret
    )
    return pyo.inequality(
        Globals.Target_Trade_Spend[Cur_Ret] * (1 - Globals.Ov_Perc_Limit / 100),
        Val,
        Globals.Target_Trade_Spend[Cur_Ret] * (1 + Globals.Ov_Perc_Limit / 100),
    )


def EDLP_Trade_Spent_Bnd_Stg1_Fn(Model, Cur_Ret):
    """Define pyomo callback for calculating bound for EDLP trade spent.

    Parameters
    ----------
    Model : Pyomo Model Object created with the required variables
    Cur_Ret :    Integer, PPG number to calculate EDLP Trade Spent for all the weeks.
    Will be iteratively called by Model Objective function
    Returns:
    --------
    Pyomo Expression, containing product EDLP Trade Spent equation

    """
    Val = (
        Model.Weeks[Cur_Ret]
        * (Globals.Base_Price_stg1[Cur_Ret] - Retailer_Price_Fn_Const1(Model, Cur_Ret))
        * Retailer_Unit_Sales_Fn_Const1(Model, Cur_Ret)
    )

    return pyo.inequality(
        Globals.Target_EDLP_Spend[Cur_Ret] * (1 - Globals.EDLP_Perc_Limit / 100),
        Val,
        Globals.Target_EDLP_Spend[Cur_Ret] * (1 + Globals.EDLP_Perc_Limit / 100),
    )


def TPR_Spend_calc_edge_case(Model, P):
    """Define pyomo callback for calculating bound for EDLP trade spent.

    Parameters
    ----------
    Model : Pyomo Model Object created with the required variables
    P :    Integer, PPG number to calculate TPR Trade Spent for all the weeks.
    Will be iteratively called by Model Objective function
    Returns:
    --------
    Pyomo Expression, containing product TPR Trade Spent equation

    """
    Self = pyo.exp(
        pyo.log(Globals.Base_Price_stg1[P]) * Globals.EDLP_Coef[P][P]
        + Globals.TPR_LB * TPR_Conv(Model, P)
        + Globals.Intercepts_stg1[P]
    )
    Comp_Retailers_Unit_Sales = [
        pyo.exp((pyo.log(Globals.Base_Price_stg1[i]) * Globals.EDLP_Coef[P][i]))
        for i in Model.PPG_index
        if i != P
    ]
    Comp = reduce(lambda x, y: x * y, Comp_Retailers_Unit_Sales, 1)
    Unit_Sales = Self * Comp
    Price = Globals.Base_Price_stg1[P] * (1 - Globals.TPR_LB)

    TPR_LB_Spend = (Globals.Base_Price_stg1[P] - Price) * Unit_Sales

    TPR_Spend_Buffer = Globals.Target_Trade_Spend[P] * (
        1 + Globals.Ov_Perc_Limit / 100
    ) - Globals.Target_EDLP_Spend[P] * (1 - Globals.EDLP_Perc_Limit / 100)

    if TPR_LB_Spend <= TPR_Spend_Buffer:
        Buffer_TPR_Events = int(TPR_Spend_Buffer / TPR_LB_Spend)
        return (Buffer_TPR_Events, TPR_Spend_Buffer)
    else:
        return 0, 0


def TPR_Trade_Spent_Bnd_Stg1_Fn(Model, Cur_Ret):
    """Define pyomo callback for calculating bound for TPR trade spent.

    Parameters
    ----------
    Model : Pyomo Model Object created with the required variables
    Cur_Ret : Integer, PPG number to calculate TPR Trade Spent for all the weeks.
    Will be iteratively called by Model Objective function
    Returns:
    --------
    Pyomo Expression, containing product TPR Trade Spent equation

    """

    Val = (
        (Globals.Tot_Week - Model.Weeks[Cur_Ret])
        * Retailer_Unit_Sales_Fn_Const2(Model, Cur_Ret)
        * Globals.TE_Val[Cur_Ret][1]
    )

    LHS = Globals.Target_TPR_Spend[Cur_Ret] * (1 - Globals.TPR_Perc_Limit / 100)
    RHS = Globals.Target_TPR_Spend[Cur_Ret] * (1 + Globals.TPR_Perc_Limit / 100)
    if RHS == 0:
        Max_Events, UB_TPR_Spend = TPR_Spend_calc_edge_case(Model, Cur_Ret)
        if UB_TPR_Spend:
            RHS = UB_TPR_Spend
    return pyo.inequality(LHS, Val, RHS)


def Overall_Total_Trade_Spent_Stg1_Fn(Model):
    """To Establish constraint on overall trade spend for pyomo model.

    Parameters
    ----------
    Model : Pyomo Model Object created with the required variables
    Returns:
    --------
    Pyomo Expression, containing OVerall Trade equation

    """
    return sum(
        [
            (
                Model.Weeks[Cur_Ret]
                * (
                    Globals.Base_Price_stg1[Cur_Ret]
                    - Retailer_Price_Fn_Const1(Model, Cur_Ret)
                )
                * Retailer_Unit_Sales_Fn_Const1(Model, Cur_Ret)
                + (Globals.Tot_Week - Model.Weeks[Cur_Ret])
                * Retailer_Unit_Sales_Fn_Const2(Model, Cur_Ret)
                * Globals.TE_Val[Cur_Ret][1]
            )
            for Cur_Ret in range(Globals.Tot_Prod)
        ]
    ) <= sum(Globals.Target_Trade_Spend) * (
        1 + Globals.Retailer_Overall_Sales_Buffer / 100
    )


def Create_Model_Stg1():
    """Create Pyomo model.

    Model consists of
    1. Functions to properly initialize number of weaks and respective bound from global data loader class
    2. Variables to optimize
    3. Objective function
    4. Constraints

    """

    def Weeks_Init(Model, PPG_index):
        return Globals.EDLP_Events[PPG_index]

    def Weeks_bound(Model, PPG_index):
        return (Globals.Min_EDLP_Events[PPG_index], Globals.Max_EDLP_Events[PPG_index])

    def TPR_Bound_Init(Model, P):

        LB = min(Globals.TPR_Perc_Val[P])
        UB = max(Globals.TPR_Perc_Val[P])

        return (LB, UB)

    def TPR_Initial(Model, P):

        UB = max(Globals.TPR_Perc_Val[P])

        return UB

    Model = pyo.ConcreteModel(name="Spend_Optim")
    Model.PPGs = pyo.Param(initialize=Globals.Tot_Prod, domain=pyo.PositiveIntegers)
    Model.PPG_index = pyo.RangeSet(0, Model.PPGs - 1)
    Model.EDLP = pyo.Var(
        Model.PPG_index,
        initialize=Globals.EDLP_UB,
        domain=pyo.NonNegativeReals,
        bounds=(Globals.EDLP_LB, Globals.EDLP_UB),
    )
    Model.TPR = pyo.Var(
        Model.PPG_index,
        initialize=TPR_Initial,
        domain=pyo.NonNegativeReals,
        bounds=TPR_Bound_Init,
    )
    Model.Weeks = pyo.Var(
        Model.PPG_index,
        initialize=Weeks_Init,
        domain=pyo.PositiveIntegers,
        bounds=Weeks_bound,
    )
    Model.Obj = pyo.Objective(rule=Dollar_Sales_Stg1_Fn, sense=pyo.maximize)
    # Model.Tot_Spent_Bnd = pyo.Constraint(Model.PPG_index, rule=Total_Trade_Spent_Bnd_Stg1_Fn)
    Model.EDLP_Bnd = pyo.Constraint(Model.PPG_index, rule=EDLP_Trade_Spent_Bnd_Stg1_Fn)
    Model.TPR_Bnd = pyo.Constraint(Model.PPG_index, rule=TPR_Trade_Spent_Bnd_Stg1_Fn)
    Model.Overall = pyo.Constraint(rule=Overall_Total_Trade_Spent_Stg1_Fn)
    return Model


def Call_Solver_Stg1(PPGs, Wks, name="bonmin", obj=1):
    """To Generate and call a pyomo model.

    The function generates a generic model which can be ran across solvers
    which are supported in the Pyomo framework and can solve MINLP problems
    ex-Bonmin, Baron.

    Parameters
    ----------
    PPG : Integer, Number of Total Products to be passed
    Wks : Integer, Number of Total Weeks to be passed
    name : String, (Optional) Name of the solver to be used
    obj : Initial value of the Objective (Optional) to keep running the solver till the maximum value is reached
    Returns:
    --------
    Pyomo Model, containing Model Object with required variables

    """
    Model = Create_Model_Stg1()

    path = get_pack_path() + "scripts/tpo_discrete/bonmin-win32/bonmin.exe"  # noqa
    Opt = pyo.SolverFactory(name, executable=path)
    start_time = time.time()
    try:
        Result = Opt.solve(Model)
        if str(Result.solver.status) != "ok":
            raise Exception("Solver Status should be OK")
        if str(Result.solver.termination_condition) != "optimal":
            raise Exception("Terminal Condition should be Optimal")
        Globals.Stage1Success = True
    except Exception as e:
        logger.error("\nError in Solving problem" + str(e))
    end_time = time.time()
    Call_Solver_Stg1.EDLP_Val = [pyo.value(Model.EDLP[i]) for i in range(PPGs)]
    Call_Solver_Stg1.TPR_Val = [pyo.value(Model.TPR[i]) for i in range(PPGs)]
    Call_Solver_Stg1.Num_Events = [pyo.value(Model.Weeks[i]) for i in range(PPGs)]
    Call_Solver_Stg1.EDLP_Spend_Values = [
        pyo.value(
            (Model.Weeks[Cur_Ret])
            * (
                Globals.Base_Price_stg1[Cur_Ret]
                - Retailer_Price_Fn_Const1(Model, Cur_Ret)
            )
            * Retailer_Unit_Sales_Fn_Const1(Model, Cur_Ret)
        )
        for Cur_Ret in range(Globals.Tot_Prod)
    ]
    Call_Solver_Stg1.TPR_Spend_Values = [
        pyo.value(
            (Globals.Tot_Week - Model.Weeks[Cur_Ret])
            * (
                Globals.Base_Price_stg1[Cur_Ret]
                - Retailer_Price_Fn_Const2(Model, Cur_Ret)
            )
            * Retailer_Unit_Sales_Fn_Const2(Model, Cur_Ret)
        )
        for Cur_Ret in range(Globals.Tot_Prod)
    ]
    Call_Solver_Stg1.Tot_Spend_Values = list(
        map(
            lambda x, y: x + y,
            Call_Solver_Stg1.EDLP_Spend_Values,
            Call_Solver_Stg1.TPR_Spend_Values,
        )
    )
    if Globals.Stage1Success and pyo.value(Model.Obj) > obj:
        logger.info(
            "\n\n Optimizer First Stage Results:\n##############################\n\n"
        )
        el_time = " The Elapsed Time is --- %s Seconds ---" % (end_time - start_time)
        btime = f"{name} Solver Execution Time: {Result['Solver'][0]['Time']}"
        logger.info(
            f"Elapsed_Time: {el_time}\n\nBonmin Time: {btime}\n\nObjective_Value: {pyo.value(Model.Obj)}"
        )
        logger.info(
            f"Message: {Result['Solver'][0]['Message']},Termination: {Result['Solver'][0]['Termination condition']},Status: {Result['Solver'][0]['Status']},Objective_Value: {pyo.value(Model.Obj)}",
        )
        Call_Solver_Stg1.Obj_Val = pyo.value(Model.Obj)
        Globals.Stage1Success = True
    else:
        Call_Solver_Stg1.Obj_Val = obj
        Globals.Stage1Success = False


def Stage1_caller(global_vars):
    """Task of the function is to get the initialized global data class.

    use the solver and load results back to the global data class thereby
    handing over control to second stage
    Parameters
    ----------
    global_vars : Object, Globally defined variables is being passed

    """
    global Globals
    Globals = global_vars

    logger.info(f"\n\nSolving First Stage : Ret {Globals.Ret} Cat {Globals.Cat}")
    Call_Solver_Stg1(PPGs=Globals.Tot_Prod, Wks=Globals.Tot_Week, name="bonmin", obj=1)
    Globals.stage1_EDLP_Init_PPG = Call_Solver_Stg1.EDLP_Val
    Globals.stage1_TPR_Init_PPG = Call_Solver_Stg1.TPR_Val
    Globals.stage1_Num_Events_PPG = Call_Solver_Stg1.Num_Events
    Globals.stage1_EDLP_Spend_Values = Call_Solver_Stg1.EDLP_Spend_Values
    Globals.stage1_TPR_Spend_Values = Call_Solver_Stg1.TPR_Spend_Values
    Globals.stage1_Tot_Spend_Values = Call_Solver_Stg1.Tot_Spend_Values


if __name__ == "__main__":
    Globals.init(1, 1)
    Stage1_caller()
