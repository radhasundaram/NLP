#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 04/06/20 10:13 PM
# @author: Gurusankar G

from __future__ import division

import logging
import time
from functools import reduce

import numpy as np
import pyomo.environ as pyo

from ta_lib.core.utils import get_package_path
from ta_lib.tpo.optimization.discrete.stage_1_tpo_optimizer import (
    Stage1_caller,
    TPR_Spend_calc_edge_case,
)

logger = logging.getLogger(__name__)


def get_pack_path():
    """Absolute path for the package."""
    return get_package_path().replace("\\", "/").replace("src", "")


def TPR_Conv(Model, P, Wk):
    """To pick relevent cutoff value for the chosen TPR value."""
    if Globals.TPR_Cutoff.iloc[P, 0] > 0:

        return Globals.TPR_Cutoff.iloc[P, 0]

    elif pyo.value(Model.TPR[P]) > Globals.TPR_Cutoff.iloc[P, 3]:

        return Globals.TPR_Cutoff.iloc[P, 2]
    else:

        return Globals.TPR_Cutoff.iloc[P, 1]


def Retailer_Unit_Sales_Stg_inter_Fn(Model, P, W):
    """Define pyomo callback for calculating product unit sales.

    Parameters
    ----------
    Model : Pyomo Model Object created with the required variables
    P :    Integer, PPG number to calculate Unit sales for the given week.
    Will be iteratively called by Model Objective function.
    Wk : Integer, Week number for the year
    Returns:
    --------
    Pyomo Expression, containing Product Unit Sales equation for the given week

    """
    Self = pyo.exp(
        (
            pyo.log(Model.EDLP[P] * Globals.Base_Price_stg2[P][W])
            * Globals.EDLP_Coef[P][P]
        )  # noqa
        * Model.FLAG[P, W]
        + (
            pyo.log(Globals.Base_Price_stg2[P][W]) * Globals.EDLP_Coef[P][P]
            + Model.TPR[P] * TPR_Conv(Model, P, W)
        )
        * (1 - Model.FLAG[P, W])
        + Globals.Intercepts_stg2[W][P]
        + (Globals.Cat_Coef[P] * Model.Catalogue_FLAG[P, W])
        + (Globals.Disp_Coef[P] * Model.Display_FLAG[P, W])
    )
    Comp = Competitor_Unit_Effect_Stg_inter_Fn(Model, P, W)
    Pantry_Loading_Val = Pantry_Loading_Unit_Effect_inter_Fn(Model, P, W)
    Unit_Sales = Self * Comp * Pantry_Loading_Val
    return Unit_Sales


def Competitor_Unit_Effect_Stg_inter_Fn(Model, Cur_Ret, Wk):
    """Define pyomo callback for calculating competitor effect.

    Parameters
    ----------
    Model : Pyomo Model Object created with the required variables
    Cur_Ret :    Integer, PPG number to calculate EDLP Competitor Effect for the given week . Will be iteratively called by Model Objective function.
    Wk : Integer, Week number for the year
    Returns:
    --------
    Pyomo Expression, containing Competitor Unit Effect for the Product for the given week

    """
    Comp_Retailers_Unit_Sales = [
        pyo.exp(
            (
                pyo.log(Model.EDLP[i] * Globals.Base_Price_stg2[i][Wk])
                * Globals.EDLP_Coef[Cur_Ret][i]
            )
            * Model.FLAG[i, Wk]
            + (
                pyo.log(Globals.Base_Price_stg2[i][Wk]) * Globals.EDLP_Coef[Cur_Ret][i]
                + Model.TPR[i] * Globals.TPR_Coef[Cur_Ret][i]
            )
            * (1 - Model.FLAG[i, Wk])
        )
        for i in Model.PPG_index
        if i != Cur_Ret
    ]
    return reduce(lambda x, y: x * y, Comp_Retailers_Unit_Sales, 1)


def Pantry_Loading_Unit_Effect_inter_Fn(Model, P, Wk):
    """Define pyomo callback for individual product pantry loading effect.

    Parameters
    ----------
    Model : Pyomo Model Object created with the required variables
    P :    Integer, PPG number to calculate Pantry Loading Effect for the given week . Will be iteratively called by Model Objective function.
    Wk : Integer, Week number for the year
    Returns:
    --------
    Pyomo Expression, containing product Pantry Unit Effect equation for the given week

    """
    Pantry_1_Effect = 1

    PL_Coef1 = Globals.Pantry_1[Wk][P]

    if Wk > 0:

        Pantry_1_Effect = pyo.exp(Model.TPR[P] * PL_Coef1 * (1 - Model.FLAG[P, Wk - 1]))

    Pantry_Loading_Unit_Effect = Pantry_1_Effect

    return Pantry_Loading_Unit_Effect


def Retailer_Price_inter_Fn(Model, P, W):
    """Define pyomo callback for individual product price with model parameters.

    Parameters
    ----------
    Model : Pyomo Model Object created with the required variables
    P :    Integer, PPG number to calculate Retailer Price for the given week . Will be iteratively called by Model Objective function.
    Wk : Integer, Week number for the year
    Returns:
    --------
    Pyomo Expression, containing product Retailer Price equation for the given week

    """
    Price = (
        Globals.Base_Price_stg2[P][W]
        * (1 - Model.TPR[P] / 100)
        * (1 - Model.FLAG[P, W])
    ) + (Model.EDLP[P] * Globals.Base_Price_stg2[P][W] * Model.FLAG[P, W])
    return Price


# ############################### Dollar Sales Stage 2 ###################################
def Dollar_Sales_Stg_inter_Fn(Model):
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
            Retailer_Unit_Sales_Stg_inter_Fn(Model, P, W)
            * Retailer_Price_inter_Fn(Model, P, W)
            for W in Model.Wk_index
            for P in Model.PPG_index
        ]
    )


def Total_Trade_Spent_Bnd_Stg_inter_Fn(Model, Cur_Ret):
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
    Val = sum(
        [
            (
                Globals.Base_Price_stg2[Cur_Ret][Wk]
                - Retailer_Price_inter_Fn(Model, Cur_Ret, Wk)
            )
            * Retailer_Unit_Sales_Stg_inter_Fn(Model, Cur_Ret, Wk)
            for Wk in Model.Wk_index
        ]
    )
    return pyo.inequality(
        Globals.Target_Trade_Spend[Cur_Ret] * (1 - Globals.Ov_Perc_Limit / 100),
        Val,
        Globals.Target_Trade_Spend[Cur_Ret] * (1 + Globals.Ov_Perc_Limit / 100),
    )


# ########################### Trade Spend Constraints #############################
def EDLP_Trade_Spent_Bnd_Stg_inter_Fn(Model, Cur_Ret):
    """Define pyomo callback for calculating bound for EDLP trade spent.

    Parameters
    ----------
    Model : Pyomo Model Object created with the required variables
    Cur_Ret :    Integer, PPG number to calculate EDLP Trade Spent for all the weeks . Will be iteratively called by Model Objective function
    Returns:
    --------
    Pyomo Expression, containing product EDLP Trade Spent equation

    """
    Val = sum(
        [
            (
                Globals.Base_Price_stg2[Cur_Ret][Wk]
                - Retailer_Price_inter_Fn(Model, Cur_Ret, Wk)
            )
            * Retailer_Unit_Sales_Stg_inter_Fn(Model, Cur_Ret, Wk)
            * (Model.FLAG[Cur_Ret, Wk])
            for Wk in Model.Wk_index
        ]
    )
    return pyo.inequality(
        Globals.Target_EDLP_Spend[Cur_Ret] * (1 - Globals.EDLP_Perc_Limit / 100),
        Val,
        Globals.Target_EDLP_Spend[Cur_Ret] * (1 + Globals.EDLP_Perc_Limit / 100),
    )


def TPR_Trade_Spent_Bnd_Stg_inter_Fn(Model, Cur_Ret):
    """Define pyomo callback for calculating bound for TPR trade spent.

    Parameters
    ----------
    Model : Pyomo Model Object created with the required variables
    Cur_Ret :    Integer, PPG number to calculate TPR Trade Spent for all the weeks . Will be iteratively called by Model Objective function
    Returns:
    --------
    Pyomo Expression, containing product TPR Trade Spent equation

    """

    Val = sum(
        [
            (Globals.TE_Val[Cur_Ret][1])
            * Retailer_Unit_Sales_Stg_inter_Fn(Model, Cur_Ret, Wk)
            * (1 - Model.FLAG[Cur_Ret, Wk])
            for Wk in Model.Wk_index
        ]
    )

    LHS = Globals.Target_TPR_Spend[Cur_Ret] * (1 - Globals.TPR_Perc_Limit / 100)
    RHS = Globals.Target_TPR_Spend[Cur_Ret] * (1 + Globals.TPR_Perc_Limit / 100)
    if RHS == 0:
        Max_Events, UB_TPR_Spend = TPR_Spend_calc_edge_case(Model, Cur_Ret)
        if Globals.Tot_Week - Globals.stage1_Num_Events_PPG[Cur_Ret] != 0:
            RHS = UB_TPR_Spend
    return pyo.inequality(LHS, Val, RHS)


def Overall_Total_Trade_Spent_Stg_inter_Fn(Model):
    """To establish constraint on overall trade spend for pyomo model.

    Parameters
    ----------
    Model : Pyomo Model Object created with the required variables
    Returns:
    --------
    Pyomo Expression, containing OVerall Trade equation

    """
    Val = sum(
        [
            sum(
                [
                    (
                        Globals.Base_Price_stg2[Cur_Ret][Wk]
                        - Retailer_Price_inter_Fn(Model, Cur_Ret, Wk)
                    )
                    * Retailer_Unit_Sales_Stg_inter_Fn(Model, Cur_Ret, Wk)
                    * (Model.FLAG[Cur_Ret, Wk])
                    + (Globals.TE_Val[Cur_Ret][1])
                    * Retailer_Unit_Sales_Stg_inter_Fn(Model, Cur_Ret, Wk)
                    * (1 - Model.FLAG[Cur_Ret, Wk])
                    for Wk in range(Globals.Tot_Week)
                ]
            )
            for Cur_Ret in range(Globals.Tot_Prod)
        ]
    )

    LHS = sum(Globals.Target_Trade_Spend)
    RHS = sum(Globals.Target_Trade_Spend) * (
        1 + Globals.Retailer_Overall_Sales_Buffer / 100
    )
    return pyo.inequality(LHS, Val, RHS)


# ############################## Events constraints #####################################
def FLAG_Util_Fn(Model, Cur_Ret):
    """Define constraint bounds callback function for pyomo model on FLAGS for given model parameter values.

    Parameters
    ----------
    Model : Pyomo Model Object created with the required variables
    Cur_Ret : Integer, PPG number of the Product
    Returns:
    --------
    Pyomo Model, containing FLAG Constraint Function

    """
    return (
        sum(
            [
                Model.FLAG[Cur_Ret, Wk] * (1 - Model.FLAG[Cur_Ret, Wk])
                for Wk in range(Globals.Tot_Week)
            ]
        )
        == 0
    )


def EDLP_Event_Fn(Model, P):
    """Define constraint bounds callback function for pyomo model on EDLP events for given model parameter values.

    Parameters
    ----------
    Model : Pyomo Model Object created with the required variables
    P : Integer, PPG number of the Product
    Returns:
    --------
    Pyomo Model, containing containing EDLP Event Constraint Function

    """
    return (
        sum([Model.FLAG[P, W] for W in Model.Wk_index])
        == Globals.stage1_Num_Events_PPG[P]
    )


def TPR_Event_Fn(Model, P):
    """Define constraint bounds callback function for pyomo model on TPR envents for given model parameter values.

    Parameters
    ----------
    Model : Pyomo Model Object created with the required variables
    P : Integer, PPG number of the Product
     Returns:
    --------
    Pyomo Model, containing containing TPR Event Constraint Function

    """
    return (
        Globals.Tot_Week - sum([Model.FLAG[P, W] for W in Model.Wk_index])
        == Globals.Tot_Week - Globals.stage1_Num_Events_PPG[P]
    )


def Same_51_52(Model, P):  # Stage2
    """Enforcing same promotion for the last two weeks for the year."""
    return Model.FLAG[P, 50] == Model.FLAG[P, 51]


def Same_TPR_Events_Half_year(Model, P):
    """Enforcing same number of Promotion events in the first half and second half of the year."""
    First_Half = [(Model.FLAG[P, Wk]) for Wk in range(0, Globals.Tot_Week // 2)]
    Second_Half = [
        (Model.FLAG[P, Wk]) for Wk in range(Globals.Tot_Week // 2, Globals.Tot_Week)
    ]
    return sum(First_Half) == sum(Second_Half)


def Weeks_Baseline(Model, run=True):
    """Enforcing certain weeks to be same as baseline calendar."""
    if run:
        Model.Weeks_Baseline_Constraint = pyo.ConstraintList()
        Weeks_Baseline = Globals.Weeks_Baseline_2

        for Prod in range(len(Weeks_Baseline)):
            for Wk in Weeks_Baseline[Prod]:
                Model.Weeks_Baseline_Constraint.add(
                    Model.FLAG[Prod, Wk] == Globals.Baseline_FLAG[Prod, Wk]
                )


def Catalogue_Constraint_Fn(Model, P):
    """Enforcing number of catalogue weeks to be within historical catalogue weeks."""

    return (
        sum([Model.Catalogue_FLAG[P, Wk] for Wk in Model.Wk_index])
        <= Globals.Cat_Sum[P]
    )


def Display_Constraint_Fn(Model, P):
    """Enforcing number of display weeks to be within historical display weeks."""

    return (
        sum([Model.Display_FLAG[P, Wk] for Wk in Model.Wk_index]) <= Globals.Disp_Sum[P]
    )


def Cat_Constraint_Fn(Model, P, Wk):
    """Enforcing catalogue weeks only on promotion weeks."""

    return Model.Catalogue_FLAG[P, Wk] <= (1 - Model.FLAG[P, Wk])


def Disp_Constraint_Fn(Model, P, Wk):
    """Enforcing display weeks only on promotion weeks."""
    return Model.Display_FLAG[P, Wk] <= (1 - Model.FLAG[P, Wk])


def Cat_Clean_Air_Constraint(Model, run=True):
    """Enforcing No promotion weeks for alteast 2 prior weeks before the catalogue week."""
    if run:
        Model.Clean_Air_Constraint = pyo.ConstraintList()
        for P in range(Globals.Tot_Prod):
            EDLP_Flag = [(Model.FLAG[P, Wk]) for Wk in Model.Wk_index]
            Catl_Flag = [(Model.Catalogue_FLAG[P, Wk]) for Wk in Model.Wk_index]
            Three_Weeks = [
                (Catl_Flag[Wk] * EDLP_Flag[Wk - 1] * EDLP_Flag[Wk - 2])
                + (1 - Catl_Flag[Wk])
                for Wk in range(2, Globals.Tot_Week)
            ]

            for expr in Three_Weeks:
                Model.Clean_Air_Constraint.add(expr == 1)


def Same_Calendar_Constraint(Model, run=True):
    """Enforcing same promotion calendar for two products."""
    if run:
        Same_Cal_PPG = Globals.Same_Cal_PPG_2
        Model.Same_Calendar_PPG = pyo.ConstraintList()

        for Wk in Model.Wk_index:
            Model.Same_Calendar_PPG.add(
                Model.FLAG[Same_Cal_PPG[0], Wk] == Model.FLAG[Same_Cal_PPG[1], Wk]
            )


def Create_Model_Stg_inter(flag_init, sec_sing_comb):
    def FLAG_Initialize(Model, *Index):
        return flag_init[Index]

    def TPR_Bound_Init(Model, P):

        LB = min(Globals.TPR_Perc_Val[P])
        UB = max(Globals.TPR_Perc_Val[P])

        return (LB, UB)

    def TPR_Initial(Model, P):

        UB = max(Globals.TPR_Perc_Val[P])

        return UB

    Model = pyo.ConcreteModel(name="Spend_Optim")
    Model.PPGs = pyo.Param(initialize=Globals.Tot_Prod, domain=pyo.PositiveIntegers)
    Model.Weeks = pyo.Param(initialize=Globals.Tot_Week, domain=pyo.PositiveIntegers)
    Model.PPG_index = pyo.RangeSet(0, Model.PPGs - 1)
    Model.Wk_index = pyo.RangeSet(0, Model.Weeks - 1)
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
    Model.FLAG = pyo.Var(
        Model.PPG_index, Model.Wk_index, initialize=FLAG_Initialize, domain=pyo.Binary
    )

    # if sec_sing_comb[1] or sec_sing_comb[2]:
    Model.Catalogue_FLAG = pyo.Var(
        Model.PPG_index, Model.Wk_index, initialize=0, domain=pyo.Binary
    )
    Model.Display_FLAG = pyo.Var(
        Model.PPG_index, Model.Wk_index, initialize=0, domain=pyo.Binary
    )

    Model.Obj = pyo.Objective(rule=Dollar_Sales_Stg_inter_Fn, sense=pyo.maximize)
    Model.TPR_Bnd = pyo.Constraint(
        Model.PPG_index, rule=TPR_Trade_Spent_Bnd_Stg_inter_Fn
    )
    Model.FLAG_Util_c = pyo.Constraint(Model.PPG_index, rule=FLAG_Util_Fn)
    Model.TPR_Event = pyo.Constraint(Model.PPG_index, rule=TPR_Event_Fn)
    Model.EDLP_Event = pyo.Constraint(Model.PPG_index, rule=EDLP_Event_Fn)
    Model.Overall = pyo.Constraint(rule=Overall_Total_Trade_Spent_Stg_inter_Fn)

    if sec_sing_comb[0]:
        Model.Same_51_52 = pyo.Constraint(Model.PPG_index, rule=Same_51_52)

    Weeks_Baseline(Model, sec_sing_comb[1])

    Model.Catalogue_Constraint = pyo.Constraint(
        Model.PPG_index, rule=Catalogue_Constraint_Fn
    )
    Model.Display_Constraint = pyo.Constraint(
        Model.PPG_index, rule=Display_Constraint_Fn
    )
    Model.Cat_Flag_Constraint_Fn = pyo.Constraint(
        Model.PPG_index, Model.Wk_index, rule=Cat_Constraint_Fn
    )
    Model.Disp_Flag_Constraint_Fn = pyo.Constraint(
        Model.PPG_index, Model.Wk_index, rule=Disp_Constraint_Fn
    )

    Cat_Clean_Air_Constraint(Model, sec_sing_comb[2])  # TODO:
    Same_Calendar_Constraint(Model, sec_sing_comb[3])

    return Model


def Call_Solver_Stg_Inter(PPGs, Wks, sec_sing_comb, name="bonmin", obj=1):
    flag_init = Globals.historical_df.Event.tolist()
    flag_init = np.array(flag_init).reshape(Globals.Tot_Prod, Globals.Tot_Week)
    Model = Create_Model_Stg_inter(flag_init, sec_sing_comb)

    path = get_pack_path() + "scripts/tpo_discrete/bonmin-win32/bonmin.exe"  # noqa
    opt = pyo.SolverFactory(name, executable=path)
    opt.options["tol"] = 1e-4
    opt.options["max_cpu_time"] = 120
    opt.options["bonmin.algorithm"] = "B-Hyb"
    start_time = time.time()
    if Globals.Stage1Success:
        try:
            Result = opt.solve(Model)
            if str(Result.solver.status) != "ok":
                raise Exception("Solver Status should be OK")
            if str(Result.solver.termination_condition) != "optimal":
                raise Exception("Terminal Condition should be Optimal")
            Globals.Stage2Success = True
        except Exception as e:
            logger.error("\nError in Solving problem" + str(e))
    end_time = time.time()
    Call_Solver_Stg_Inter.EDLP_Val = [pyo.value(Model.EDLP[i]) for i in range(PPGs)]
    Call_Solver_Stg_Inter.TPR_Val = [pyo.value(Model.TPR[i]) for i in range(PPGs)]
    Call_Solver_Stg_Inter.FLAG = [
        [pyo.value(Model.FLAG[i, W]) for W in Model.Wk_index] for i in range(PPGs)
    ]

    Call_Solver_Stg_Inter.Catalogue_FLAG_Val = [
        [pyo.value(Model.Catalogue_FLAG[i, W]) for W in Model.Wk_index]
        for i in range(PPGs)
    ]
    Call_Solver_Stg_Inter.Display_FLAG_Val = [
        [pyo.value(Model.Display_FLAG[i, W]) for W in Model.Wk_index]
        for i in range(PPGs)
    ]

    if Globals.Stage2Success and pyo.value(Model.Obj) > obj:
        logger.info(
            "\n\n Optimizer Second Stage Results:\n##############################\n\n"
        )
        #         log_infeasible_constraints(Model)
        #         log_infeasible_bounds(Model)
        logger.info(pyo.value(Model.Obj))
        el_time = " The Elapsed Time is --- %s Seconds ---" % (end_time - start_time)
        btime = f"{name} Solver Execution Time: {Result['Solver'][0]['Time']}"
        logger.info(
            f"Elapsed_Time: {el_time}\n\nBonmin Time: {btime}\n\nObjective_Value: {pyo.value(Model.Obj)}"
        )
        logger.info(
            f"Message: {Result['Solver'][0]['Message']},Termination: {Result['Solver'][0]['Termination condition']},Status: {Result['Solver'][0]['Status']},Objective_Value: {pyo.value(Model.Obj)}",
        )
        Call_Solver_Stg_Inter.Obj_Val = pyo.value(Model.Obj)
        Globals.Stage2Success = True
    else:
        Call_Solver_Stg_Inter.Obj_Val = obj
        Globals.Stage2Success = False


def Stage2_caller(global_vars, sec_sing_comb):
    """Task of the function is to get the initialized global data class.

    Inject class into first stage and wait for the outputs form the first stage solver. Then it shall
    call the second stage of the pipeline with interim FLAG values to be used as init values
    in this phase. Loaded result within the global data class is then passed on to the third stage
    Parameters
    ----------
    global_vars : Object, Globally defined variables is being passed

    """
    global Globals
    Globals = global_vars
    Stage1_caller(global_vars)
    logger.info(f"\n\nSolving Second Stage : Ret {Globals.Ret} Cat {Globals.Cat}")
    Call_Solver_Stg_Inter(
        PPGs=Globals.Tot_Prod,
        Wks=Globals.Tot_Week,
        sec_sing_comb=sec_sing_comb,
        name="bonmin",
        obj=1,
    )
    Globals.FLAG_Inter = np.array(Call_Solver_Stg_Inter.FLAG)
    Globals.TPR_Inter = Call_Solver_Stg_Inter.TPR_Val
    Globals.EDLP_Inter = Call_Solver_Stg_Inter.EDLP_Val

    Globals.Cat_FLAG_Val = np.array(Call_Solver_Stg_Inter.Catalogue_FLAG_Val)
    Globals.Disp_FLAG_Val = np.array(Call_Solver_Stg_Inter.Display_FLAG_Val)


if __name__ == "__main__":
    Globals.init(0, 0)
    Stage2_caller()
