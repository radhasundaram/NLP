#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 04/06/20 05:22 PM
# @author: Gurusankar G

from __future__ import division

import logging
import time
from functools import reduce

import numpy as np
import pyomo.environ as pyo

from ta_lib.core.utils import get_package_path
from ta_lib.tpo.optimization.discrete.stage_2_tpo_optimizer import (
    Stage2_caller,
    TPR_Spend_calc_edge_case,
)


def get_pack_path():
    """Absolute path for the package."""
    return get_package_path().replace("\\", "/").replace("src", "")


logger = logging.getLogger(__name__)


class Vars:
    """To Create dynamic variables according to the number of promos."""

    pass


GLV = Vars()


def TPR_Conv(TPR_val, P, Wk):
    """To pick relevent cutoff value for the chosen TPR value."""
    if Globals.TPR_Cutoff.iloc[P, 0] > 0:

        return Globals.TPR_Cutoff.iloc[P, 0]

    elif pyo.value(TPR_val) > Globals.TPR_Cutoff.iloc[P, 3]:

        return Globals.TPR_Cutoff.iloc[P, 2]
    else:

        return Globals.TPR_Cutoff.iloc[P, 1]


def get_TPR_value(Model, P, W):
    """Return the Pyomo tpr value expression."""
    TPF = getattr(GLV.Model, f"TPR_FLAG_PPG_{P}")
    PI = getattr(GLV.Model, f"Promo_index_{P}")

    Val = [TPF[W, promo] * Globals.TPR_Perc_Val[P][promo] for promo in PI]  # noqa
    return sum(Val)


def get_TPR_eqn(Model, P, W):
    """Return the Pyomo tpr equation expression."""
    TPF = getattr(GLV.Model, f"TPR_FLAG_PPG_{P}")
    PI = getattr(GLV.Model, f"Promo_index_{P}")

    Val = [
        TPF[W, promo]
        * Globals.TPR_Perc_Val[P][promo]
        * TPR_Conv(Globals.TPR_Perc_Val[P][promo], P, W)
        for promo in PI
    ]

    return sum(Val)


def get_TPR_spend_coeff(Model, P, W):
    """Return the Pyomo tpr spend coefficient expression."""
    TPF = getattr(GLV.Model, f"TPR_FLAG_PPG_{P}")
    PI = getattr(GLV.Model, f"Promo_index_{P}")

    Val = [TPF[W, promo] * Globals.TE_Val[P][promo + 1] for promo in PI]
    return sum(Val)


def Retailer_Unit_Sales_Stg3_Fn(Model, P, W):
    """Define pyomo callback for calculating product unit sales.

    Parameters
    ----------
    Model : Pyomo Model Object created with the required variables
    P :    Integer, PPG number to calculate Unit sales for the given week . Will be iteratively called by Model Objective function
    Wk : Integer, Week number for the year
    Returns:
    --------
    Pyomo Expression, containing Product Unit Sales equation for the given week

    """
    Self = pyo.exp(
        (
            pyo.log(Model.EDLP[P, W] * Globals.Base_Price_stg2[P][W])
            * Globals.EDLP_Coef[P][P]
        )
        * Globals.FLAG_Inter[P, W]
        + (
            pyo.log(Globals.Base_Price_stg2[P][W]) * Globals.EDLP_Coef[P][P]
            + get_TPR_eqn(Model, P, W)
        )
        * (1 - Globals.FLAG_Inter[P, W])
        + Globals.Intercepts_stg2[W][P]
        + (Globals.Cat_Coef[P] * Globals.Cat_FLAG_Val[P, W])
        + (Globals.Disp_Coef[P] * Globals.Disp_FLAG_Val[P, W])
    )

    Comp = Competitor_Unit_Effect_Stg3_Fn(Model, P, W)

    Pantry_Loading_Val = Pantry_Loading_Unit_Effect_Fn(Model, P, W)
    Unit_Sales = Self * Comp * Pantry_Loading_Val

    return Unit_Sales


def Competitor_Unit_Effect_Stg3_Fn(Model, Cur_Ret, Wk):
    """Define pyomo callback for calculating competitor effect.

    Parameters
    ----------
    Model : Pyomo Model Object created with the required variables
    Cur_Ret :    Integer, PPG number to calculate EDLP Competitor Effect for the given week . Will be iteratively called by Model Objective function
    Wk : Integer, Week number for the year
    Returns:
    --------
    Pyomo Expression, containing Competitor Unit Effect for the Product for the given week

    """
    Comp_Retailers_Unit_Sales = [
        pyo.exp(
            (
                pyo.log(Model.EDLP[i, Wk] * Globals.Base_Price_stg2[i][Wk])
                * Globals.EDLP_Coef[Cur_Ret][i]
            )
            * Globals.FLAG_Inter[i, Wk]
            + (
                pyo.log(Globals.Base_Price_stg2[i][Wk]) * Globals.EDLP_Coef[Cur_Ret][i]
                + get_TPR_value(Model, i, Wk) * Globals.TPR_Coef[Cur_Ret][i]
            )
            * (1 - Globals.FLAG_Inter[i, Wk])
        )
        for i in Model.PPG_index
        if i != Cur_Ret
    ]
    return reduce(lambda x, y: x * y, Comp_Retailers_Unit_Sales, 1)
    # return 1


def Pantry_Loading_Unit_Effect_Fn(Model, P, Wk):
    """Define pyomo callback for individual product pantry loading effect.

    Parameters
    ----------
    Model : Pyomo Model Object created with the required variables
    P :    Integer, PPG number to calculate Pantry Loading Effect for the given week . Will be iteratively called by Model Objective function
    Wk : Integer, Week number for the year
    Returns:
    --------
    Pyomo Expression, containing product Pantry Unit Effect equation for the given week

    """
    Pantry_1_Effect = 1
    PL_Coef1 = Globals.Pantry_1[Wk][P]

    if Wk > 0:
        Pantry_1_Effect = pyo.exp(
            (get_TPR_value(Model, P, Wk - 1))
            * PL_Coef1
            * (1 - Globals.FLAG_Inter[P, Wk - 1])
        )

    Pantry_Loading_Unit_Effect = Pantry_1_Effect

    return Pantry_Loading_Unit_Effect


def Retailer_Price_Fn(Model, P, W):
    """Define pyomo callback for individual product price with model parameters.

    Parameters
    ----------
    Model : Pyomo Model Object created with the required variables
    P :    Integer, PPG number to calculate Retailer Price for the given week . Will be iteratively called by Model Objective function
    Wk : Integer, Week number for the year
    Returns:
    --------
    Pyomo Expression, containing product Retailer Price equation for the given week

    """
    Price = (
        Globals.Base_Price_stg2[P][W]
        * (1 - (get_TPR_value(Model, P, W)) / 100)
        * (1 - Globals.FLAG_Inter[P, W])
    ) + (Model.EDLP[P, W] * Globals.Base_Price_stg2[P][W] * Globals.FLAG_Inter[P, W])
    return Price


def Dollar_Sales_Fn(Model):
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
            Retailer_Unit_Sales_Stg3_Fn(Model, P, W) * Retailer_Price_Fn(Model, P, W)
            for W in Model.Wk_index
            for P in Model.PPG_index
        ]
    )


# ########################### Trade Spend Constraints ###########################
def Total_Trade_Spent_Bnd_Fn(Model, Cur_Ret):
    """Define pyomo callback for calculating bound for total trade spent.

    Parameters
    ----------
    Model : Pyomo Model Object created with the required variables
    Cur_Ret :    Integer, PPG number to calculate Total Trade Spent for all the weeks . Will be iteratively called by Model Objective function
    Returns:
    --------
    Pyomo Expression, containing product Total Trade Spent equation

    """
    Val = sum(
        [
            (
                Globals.Base_Price_stg2[Cur_Ret][Wk]
                - Retailer_Price_Fn(Model, Cur_Ret, Wk)
            )
            * Retailer_Unit_Sales_Stg3_Fn(Model, Cur_Ret, Wk)
            for Wk in Model.Wk_index
        ]
    )
    return pyo.inequality(
        Globals.Target_Trade_Spend[Cur_Ret] * (1 - Globals.Ov_Perc_Limit / 100),
        Val,
        Globals.Target_Trade_Spend[Cur_Ret] * (1 + Globals.Ov_Perc_Limit / 100),
    )


def TPR_Trade_Spent_Bnd_Fn(Model, Cur_Ret):
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
            get_TPR_spend_coeff(Model, Cur_Ret, Wk)
            * Retailer_Unit_Sales_Stg3_Fn(Model, Cur_Ret, Wk)
            * (1 - Globals.FLAG_Inter[Cur_Ret, Wk])
            for Wk in range(Globals.Tot_Week)
        ]
    )
    LHS = Globals.Target_TPR_Spend[Cur_Ret] * (1 - Globals.TPR_Perc_Limit / 100)
    RHS = Globals.Target_TPR_Spend[Cur_Ret] * (1 + Globals.TPR_Perc_Limit / 100)
    if RHS == 0:
        Max_Events, UB_TPR_Spend = TPR_Spend_calc_edge_case(Model, Cur_Ret)
        if UB_TPR_Spend:
            # []
            RHS = UB_TPR_Spend

    return pyo.inequality(LHS, Val, RHS)


def Overall_Total_Trade_Spent_Fn(Model):
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
                    get_TPR_spend_coeff(Model, Cur_Ret, Wk)
                    * (1 - Globals.FLAG_Inter[Cur_Ret, Wk])
                    * Retailer_Unit_Sales_Stg3_Fn(Model, Cur_Ret, Wk)
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


def Promo_constraint(Model, P, W):
    """Enforcing only one promotion per week."""
    TPF = getattr(GLV.Model, f"TPR_FLAG_PPG_{P}")
    PI = getattr(GLV.Model, f"Promo_index_{P}")

    Val = [TPF[W, Flag] for Flag in PI]
    return sum(Val) + Globals.FLAG_Inter[P, W] == 1


def Constraint_51_52(Model, run=True):
    """Enforcing same promotion for the last two weeks for the year."""
    if run:
        Model.Same_Promo_Weeks_51_52 = pyo.ConstraintList()
        for P in range(Globals.Tot_Prod):
            TPF = getattr(GLV.Model, f"TPR_FLAG_PPG_{P}")
            PI = getattr(GLV.Model, f"Promo_index_{P}")

            for Promo in PI:
                Model.Same_Promo_Weeks_51_52.add(TPF[50, Promo] == TPF[51, Promo])


def Promo_limits(Model, run=True):
    """Enforcing promotion distribution between lower and upper bounds."""
    if run:
        Model.Promo_Limits_Constraint = pyo.ConstraintList()
        LB_UB_Limits = Globals.LB_UB_Limits_3
        for Prd in range(Globals.Tot_Prod):
            TPF = getattr(GLV.Model, f"TPR_FLAG_PPG_{Prd}")  # noqa

            for Promo in range(1, len(LB_UB_Limits[Prd]) + 1):

                Val = sum([TPF[Wk, Promo - 1] for Wk in range(Globals.Tot_Week)])
                Model.Promo_Limits_Constraint.add(
                    pyo.inequality(
                        LB_UB_Limits[Prd][Promo][0], Val, LB_UB_Limits[Prd][Promo][1]
                    )
                )


def Promo_Consective(Model, run=True):
    """Enforcing ceratain promotion should not repeat in consecutive weeks."""
    if run:
        Model.Promo_Consective_Constraints = pyo.ConstraintList()
        Promo_Wk_Limit = Globals.Promo_Wk_Limit_3
        for P in range(Globals.Tot_Prod):
            TPF = getattr(GLV.Model, f"TPR_FLAG_PPG_{P}")  # noqa
            PI = getattr(GLV.Model, f"Promo_index_{P}")  # noqa

            for Promo in PI:
                PPG_Promo_Flag = [TPF[Wk, Promo] for Wk in range(Globals.Tot_Week)]
                Consecutive_Rolling_Window = [
                    sum(PPG_Promo_Flag[i : i + Promo_Wk_Limit[P] + 1])  # noqa
                    for i in range(Globals.Tot_Week - Promo_Wk_Limit[P])
                ]

                for i in Consecutive_Rolling_Window:
                    Model.Promo_Consective_Constraints.add(i <= Promo_Wk_Limit[P])


def Deep_Promo_Repeat(Model, run=True, window_size=4):
    """Enforcing Deep promotion can repeat only after window_size weeks."""
    if run:
        Promo_rep_Wk_Limit = Globals.Promo_rep_Wk_Limit_3
        Model.Deep_Promo_Rep_Constraints = pyo.ConstraintList()
        for Prod, Promo in Promo_rep_Wk_Limit.items():
            TPF = getattr(GLV.Model, f"TPR_FLAG_PPG_{Prod}")
            Deep_Promo_Flag = [TPF[Wk, Promo] for Wk in range(Globals.Tot_Week)]
            Consecutive_Rolling_Window = [
                sum(Deep_Promo_Flag[i : i + window_size])
                for i in range(Globals.Tot_Week - 3)
            ]

            for i in Consecutive_Rolling_Window:
                Model.Deep_Promo_Rep_Constraints.add(i <= 1)


def Deep_Promo_Collective(Model, run=True):
    """Enforcing No promotions in a collective should repeat only in n number of weeks."""
    if run:
        Deep_Promos_Collective = Globals.Deep_Promos_Collective_3
        DPC_nweeks = Globals.DPC_nweeks_3
        Model.Deep_Promo_Collective_Constraints = pyo.ConstraintList()
        for P in range(Globals.Tot_Prod):
            TPF = getattr(GLV.Model, f"TPR_FLAG_PPG_{P}")  # noqa

            for Promo in Deep_Promos_Collective[P]:
                DPC_Flag = [TPF[Wk, Promo] for Wk in range(Globals.Tot_Week)]
                Consecutive_Rolling_Window = [
                    sum(DPC_Flag[i : i + DPC_nweeks[P] + 1])
                    for i in range(Globals.Tot_Week - DPC_nweeks[P])
                ]

                for i in Consecutive_Rolling_Window:
                    Model.Deep_Promo_Collective_Constraints.add(i <= DPC_nweeks[P])


def Promos_Rep_PR(Model, run=True):
    """Enforcing No promotion can repeat more than n weeks in a row."""
    if run:
        Model.Promos_Rep_PR_Constraints = pyo.ConstraintList()

        Promos_Rep_PR = Globals.Promos_Rep_PR_3
        PR_nweeks = Globals.PR_nweeks_3

        for Prod in range(len(Promos_Rep_PR)):
            TPF = getattr(GLV.Model, f"TPR_FLAG_PPG_{Prod}")

            for Promo in Promos_Rep_PR[Prod]:
                Promos_Rep_PR_Flag = [
                    TPF[Wk, Promo - 1] for Wk in range(Globals.Tot_Week)
                ]
                Consecutive_Rolling_Window = [
                    sum(Promos_Rep_PR_Flag[i : i + PR_nweeks[Prod] + 1])
                    for i in range(Globals.Tot_Week - PR_nweeks[Prod])
                ]

                for i in Consecutive_Rolling_Window:
                    Model.Promos_Rep_PR_Constraints.add(i <= PR_nweeks[Prod])


def Weeks_Baseline(Model, run=True):
    """Enforcing certain weeks to be same as baseline calendar."""
    if run:
        Model.Weeks_Baseline_Constraint = pyo.ConstraintList()
        Weeks_Baseline = Globals.Weeks_Baseline_3

        for Prod in range(len(Weeks_Baseline)):
            TPF = getattr(GLV.Model, f"TPR_FLAG_PPG_{Prod}")

            for Wk in Weeks_Baseline[Prod]:
                if Globals.FLAG_Inter[Prod, Wk] == 0:
                    Model.Weeks_Baseline_Constraint.add(
                        TPF[Wk, Globals.Baseline_Promo[Prod, Wk]] == 1
                    )


def Promo_Retailer_NFL(Model, run=True):
    """Enforcing certain promotions not to follow each other."""
    if run:
        Promos_Nos = Globals.Promos_Nos_3
        Model.Promo_Retailer_NFL_Constraints = pyo.ConstraintList()

        from itertools import combinations as cmb

        for Prod in range(len(Promos_Nos)):
            TPF = getattr(GLV.Model, f"TPR_FLAG_PPG_{Prod}")

            for comb in cmb(Promos_Nos[Prod], 2):
                for Promo in range(len(comb) - 1):
                    #
                    for Wk in range(Globals.Tot_Week - 1):
                        Model.Promo_Retailer_NFL_Constraints.add(
                            (
                                TPF[Wk, comb[Promo] - 1]
                                * TPF[Wk + 1, comb[Promo + 1] - 1]
                            )
                            + (
                                TPF[Wk, comb[Promo + 1] - 1]
                                * TPF[Wk + 1, comb[Promo] - 1]
                            )
                            == 0
                        )


def TWP_Promo(Model, run=True):
    """Enforcing that catalogue and display weeks have atleast 20 percentage discount promotions."""
    if run:
        Model.TWP_Promo_Constraint = pyo.ConstraintList()
        TWP_Wk = []
        Value = Globals.Cat_FLAG_Val + Globals.Disp_FLAG_Val

        for Prod in range(Globals.Tot_Prod):
            TPF = getattr(GLV.Model, f"TPR_FLAG_PPG_{Prod}")
            tmp = []
            for ind, val in enumerate(Value[Prod]):
                if val >= 1:
                    tmp.append(ind)
            TWP_Wk.append(tmp)

            for Promo in [
                i for i, j in enumerate(Globals.TPR_Perc_Val[Prod]) if j < 20
            ]:
                for Wk in TWP_Wk[Prod]:
                    Model.TWP_Promo_Constraint.add(TPF[Wk, Promo] == 0)


def RSV_Dollar_Sales(Model, run=True):
    """Enforcing that Display weeks should have the minimum sales of given value."""
    if run:
        RSV_Cons_val = Globals.RSV_Cons_val_3
        Model.RSV_Dollar_Sales_Constraint = pyo.ConstraintList()
        Disp_Value = Globals.Disp_FLAG_Val
        RSV_Wk = []

        for P in range(Globals.Tot_Prod):
            tmp = []
            for ind, val in enumerate(Disp_Value[P]):
                if val >= 1:
                    tmp.append(ind)

            RSV_Wk.append(tmp)
            for W in RSV_Wk[P]:
                Model.RSV_Dollar_Sales_Constraint.add(
                    Retailer_Unit_Sales_Stg3_Fn(Model, P, W)
                    * Retailer_Price_Fn(Model, P, W)
                    >= RSV_Cons_val[P]
                )


def Cross_Ret_Clash_Constraint(Model, run=True):
    """Enforcing minimum number of same promotion weeks between two products."""
    if run:
        Model.Cross_Ret_Promo_Constraint = pyo.ConstraintList()
        Clash_PPG = Globals.Clash_PPG_3

        TPF1 = getattr(GLV.Model, f"TPR_FLAG_PPG_{Clash_PPG['PPG'][0]}")
        TPF2 = getattr(GLV.Model, f"TPR_FLAG_PPG_{Clash_PPG['PPG'][1]}")

        Promo_Difference = [
            (TPF1[Wk, Clash_PPG["Promo"][0]] * TPF2[Wk, Clash_PPG["Promo"][1]])
            for Wk in range(Globals.Tot_Week)
        ]
        Model.Cross_Ret_Promo_Constraint.add(
            sum(Promo_Difference) <= Clash_PPG["Upper_Clash"]
        )


def Same_Calendar_s3_Constraint(Model, run=True):
    """Enforcing same promotion calendar for two products."""
    if run:
        Same_Cal_PPG = Globals.Same_Cal_PPG_3

        TPF1 = getattr(GLV.Model, f"TPR_FLAG_PPG_{Same_Cal_PPG[0]}")
        TPF2 = getattr(GLV.Model, f"TPR_FLAG_PPG_{Same_Cal_PPG[1]}")
        PI = getattr(GLV.Model, f"Promo_index_{Same_Cal_PPG[0]}")

        Model.Same_Calendar_PPG = pyo.ConstraintList()

        for Wk in Model.Wk_index:
            for Promo in PI:
                Model.Same_Calendar_PPG.add(TPF1[Wk, Promo] == TPF2[Wk, Promo])


def Deep_Activity_Promo_Const_Fn(Model, run=True):
    """Enforcing the product not to have promotional weeks in the prior 2 weeks of the other product's promotional week."""
    if run:
        Model.Deep_Activity_Promo_Constraint = pyo.ConstraintList()
        Deep_Act = Globals.Deep_Act_3
        P0 = Deep_Act["Promo"][0]
        P1 = Deep_Act["Promo"][1]

        PPG0 = getattr(GLV.Model, f"TPR_FLAG_PPG_{Deep_Act['PPG'][0]}")
        PPG1 = getattr(GLV.Model, f"TPR_FLAG_PPG_{Deep_Act['PPG'][1]}")

        Deep_Act_Weeks_0 = [
            PPG1[Wk, P1] * (PPG0[Wk, P0] + PPG0[Wk - 1, P0] + PPG0[Wk - 2, P0])
            for Wk in range(2, Globals.Tot_Week)
        ]
        Deep_Act_Weeks_1 = [
            PPG0[Wk, P0] * (PPG1[Wk, P1] + PPG1[Wk - 1, P1] + PPG1[Wk - 2, P1])
            for Wk in range(2, Globals.Tot_Week)
        ]

        for ind in range(Globals.Tot_Week - 2):
            Model.Deep_Activity_Promo_Constraint.add(Deep_Act_Weeks_0[ind] == 0)
            Model.Deep_Activity_Promo_Constraint.add(Deep_Act_Weeks_1[ind] == 0)


# ########################## Creating Pyomo Model ###################################


def Create_Model_Stg3(EDLP_Init, TPR_Init, single_comb):
    def EDLP_Initialize(Model, *Index):
        # return EDLP_Init[Index[0]]
        return 1

    # def TPR_Initialize(Model, *Index):
    #    return TPR_Init[Index[0]]
    class Temp:
        global PROMO_Initialize

        def PROMO_Initialize(Model, w, promo):
            # promo_init = Globals.historical_df.Promo_Events.tolist()
            # #flag_init = [(1 - x) for x in flag_init]
            # promo_init = np.array(promo_init).reshape(Globals.Tot_Prod, Globals.Tot_Week)
            # if promo_init[p,w]!=0:
            #     if promo==(promo_init[p,w]-1):
            #         return 1
            #     return 0
            # else:
            #     if promo==3:
            #         return 1
            #     return 0
            # if promo==0:
            # return 1
            return 0

    GLV.Model = pyo.ConcreteModel(name="Spend_Optim")
    GLV.Model.Weeks = pyo.Param(
        initialize=Globals.Tot_Week, domain=pyo.PositiveIntegers
    )
    GLV.Model.PPGs = pyo.Param(initialize=Globals.Tot_Prod, domain=pyo.PositiveIntegers)
    GLV.Model.Wk_index = pyo.RangeSet(0, GLV.Model.Weeks - 1)
    GLV.Model.PPG_index = pyo.RangeSet(0, GLV.Model.PPGs - 1)
    # Model.EDLP = pyo.Var(Model.PPG_index, Model.Wk_index, initialize=EDLP_Initialize, domain=pyo.PositiveIntegers,
    # bounds=(Globals.EDLP_LB, Globals.EDLP_UB))
    GLV.Model.EDLP = pyo.Param(
        GLV.Model.PPG_index,
        GLV.Model.Wk_index,
        initialize=EDLP_Initialize,
        domain=pyo.PositiveIntegers,
    )
    # GLV.Model.TPR = pyo.Var(GLV.Model.PPG_index, GLV.Model.Wk_index, initialize=Globals.TPR_LB, domain=pyo.PositiveIntegers,
    #                 bounds=(Globals.TPR_LB, Globals.TPR_UB))
    for prd in range(Globals.Tot_Prod):
        setattr(
            GLV.Model,
            f"Promos_PPG_{prd}",
            pyo.Param(
                initialize=len(Globals.TPR_Perc_Val[prd]), domain=pyo.PositiveIntegers
            ),
        )
        setattr(
            GLV.Model,
            f"Promo_index_{prd}",
            pyo.RangeSet(0, getattr(GLV.Model, f"Promos_PPG_{0}") - 1),
        )
        setattr(
            GLV.Model,
            f"TPR_FLAG_PPG_{prd}",
            pyo.Var(
                GLV.Model.Wk_index,
                getattr(GLV.Model, f"Promo_index_{prd}"),
                initialize=PROMO_Initialize,
                domain=pyo.Binary,
            ),
        )

    # GLV.Model.display()

    #     GLV.Model.TPR_FLAG = pyo.Var(GLV.Model.PPG_index,GLV.Model.Wk_index,GLV.Model.Promo_index,initialize=PROMO_Initialize,domain=pyo.Binary)
    GLV.Model.Obj = pyo.Objective(rule=Dollar_Sales_Fn, sense=pyo.maximize)
    #     GLV.Model.Tot_Spent_Bnd = pyo.Constraint(GLV.Model.PPG_index, rule=Total_Trade_Spent_Bnd_Fn)
    #     GLV.Model.EDLP_Bnd = pyo.Constraint(GLV.Model.PPG_index, rule=EDLP_Trade_Spent_Bnd_Fn)
    GLV.Model.TPR_Bnd = pyo.Constraint(GLV.Model.PPG_index, rule=TPR_Trade_Spent_Bnd_Fn)
    GLV.Model.Overall = pyo.Constraint(rule=Overall_Total_Trade_Spent_Fn)
    GLV.Model.Promo_con = pyo.Constraint(
        GLV.Model.PPG_index, GLV.Model.Wk_index, rule=Promo_constraint
    )
    # Model.display()
    const_list = [
        "51_52",
        "Promo_limits",
        "Promo_Consecutive",
        "Deep_Promo_Repeat",
        "Deep_Promo_Collective",
        "Promos_Rep_PR",
        "Weeks_Baseline",
        "Promo_Retailer_NFL",
        "TWP_Promo",
        "RSV_Constraint",
        "Cross_Ret_Clash_Constraint",
        "Cat_Clean_Air_Constraint",
        "Two_Calendar_Constraint",
        "Deep_Activity_Promo_Constraint",
    ]
    name_file = ""
    # single_comb = [True,True,True,True,True,False,False,True,False,True]
    for i, s in enumerate(single_comb):
        if s:
            name_file += const_list[i] + "_"
    #     logger.info("#####------------ACK_PIN---------------###", name_file)
    #     logger.info(single_comb)

    window_size = 4

    Constraint_51_52(GLV.Model, single_comb[0])
    Promo_limits(GLV.Model, single_comb[1])
    Promo_Consective(GLV.Model, single_comb[2])
    Deep_Promo_Repeat(GLV.Model, single_comb[3], window_size=window_size)
    # Deep_Promo_Repeat(Model,single_comb[4],window_size=4)
    Deep_Promo_Collective(GLV.Model, single_comb[4])
    Promos_Rep_PR(GLV.Model, single_comb[5])
    Weeks_Baseline(GLV.Model, single_comb[6])
    Promo_Retailer_NFL(GLV.Model, single_comb[7])
    TWP_Promo(GLV.Model, single_comb[8])
    RSV_Dollar_Sales(GLV.Model, single_comb[9])
    Cross_Ret_Clash_Constraint(GLV.Model, single_comb[10])
    Same_Calendar_s3_Constraint(GLV.Model, single_comb[12])
    Deep_Activity_Promo_Const_Fn(GLV.Model, single_comb[13])

    # Model.display()
    return GLV.Model


# ########################## Solver Calling With Pyomo Model ########################


def Call_Solver_Stg3(EDLP_Init, TPR_Init, PPGs, Wks, single_comb, name="bonmin", obj=1):
    #     from pyomo.util.infeasible import log_infeasible_constraints, log_infeasible_bounds
    Model = Create_Model_Stg3(EDLP_Init, TPR_Init, single_comb)
    path = get_pack_path() + "scripts/tpo_discrete/bonmin-win32/bonmin.exe"  # noqa

    Opt = pyo.SolverFactory(name, executable=path)
    Opt.options["tol"] = 1e-2
    Opt.options["max_cpu_time"] = 300
    # opt.options['max_iter']= 50000
    Opt.options["bonmin.time_limit"] = 2000
    # Opt.options['bonmin.algorithm'] = 'B-Hyb'
    start_time = time.time()
    if Globals.Stage2Success:
        try:
            Result = Opt.solve(Model)
            if str(Result.solver.status) != "ok":
                raise Exception("Solver Status should be OK")
            if str(Result.solver.termination_condition) != "optimal":
                raise Exception("Terminal Condition should be Optimal")
            Globals.Stage3Success = True
        except Exception as e:
            logger.error("\nError in Solving problem" + str(e))
            Globals.Stage3Success = False
    end_time = time.time()
    #     Model.display()
    EDLP_Val = np.array(
        [pyo.value(Model.EDLP[i, j]) for i in range(PPGs) for j in range(Wks)]
    ).reshape(PPGs, Wks)
    TPR_Val = np.array(
        [
            pyo.value((get_TPR_value(Model, i, j)))
            for i in range(PPGs)
            for j in range(Wks)
        ]
    ).reshape(PPGs, Wks)

    if Globals.Stage3Success and pyo.value(Model.Obj) > obj:
        logger.info(
            "\n\n Optimizer Third Stage Results:\n##############################\n\n"
        )
        Globals.Solver_Model_Result = Model
        Globals.EDLP_Final = EDLP_Val
        Globals.TPR_Final = TPR_Val
        el_time = " The Elapsed Time is --- %s Seconds ---" % (end_time - start_time)
        btime = f"{name} Solver Execution Time: {Result['Solver'][0]['Time']}"
        logger.info(
            f"Elapsed_Time: {el_time}\n\nBonmin Time: {btime}\n\nObjective_Value: {pyo.value(Model.Obj)}"
        )
        logger.info(
            f"Message: {Result['Solver'][0]['Message']},Termination: {Result['Solver'][0]['Termination condition']},Status: {Result['Solver'][0]['Status']},Objective_Value: {pyo.value(Model.Obj)}",
        )
        Call_Solver_Stg3.Obj_Val = pyo.value(Model.Obj)
    else:
        #         log_infeasible_constraints(Model)
        #         log_infeasible_bounds(Model)
        Call_Solver_Stg3.Obj_Val = obj
        Globals.Solver_Model_Result = Model  # TODO: TEMP
        Globals.EDLP_Final = EDLP_Val  # TODO: TEMP
        Globals.TPR_Final = TPR_Val  # TODO: TEMP
        Globals.Stage3Success = True  # TODO: TEMP


def Stage3_caller(global_vars, single_comb, sec_sing_comb):
    """Task of the function is to get the initialized global data class.

    Inject it into second stage and wait for the outputs
    form the second stage solver then shall call the third stage solver of the pipeline with
    interim FLAG EDLP and TPR values to be used as init values in this phase
    If solution is not achieved with the above approach we move to a fallback logic
    Parameters
    ----------
    global_vars : Object, Globally defined variables is being passed

    """
    global Globals
    Globals = global_vars
    Stage2_caller(global_vars, sec_sing_comb)
    logger.info(f"\n\nSolving Third Stage : Ret {Globals.Ret} Cat {Globals.Cat}")
    Call_Solver_Stg3(
        Globals.EDLP_Inter,
        Globals.TPR_Inter,
        Globals.Tot_Prod,
        Globals.Tot_Week,
        single_comb,
        name="bonmin",
        obj=1,
    )


if __name__ == "__main__":
    Globals.init(1, 1)
    Globals.Stage2Success = True
    # Stage2_caller()
    # flag_init = Globals.historical_df.Event.tolist()
    flag_init = np.array(
        [
            [
                1,
                1,
                0,
                0,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                1,
            ],
            [
                1,
                1,
                0,
                0,
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
                1,
                0,
                0,
                1,
                0,
                1,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                1,
                1,
                1,
            ],
            [
                1,
                0,
                1,
                0,
                1,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                1,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                1,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
            ],
        ]
    )
    # flag_init = [(1 - x) for x in flag_init]
    Globals.FLAG_Inter = np.array(flag_init).reshape(Globals.Tot_Prod, Globals.Tot_Week)
    Globals.Cat_FLAG_Val = np.array([[1, 0, 0, 1] * 13] * 3)
    Globals.Disp_FLAG_Val = np.array([[1, 0, 0, 1] * 13] * 3)
    Call_Solver_Stg3(
        Globals.EDLP_Inter,
        Globals.TPR_Inter,
        Globals.Tot_Prod,
        Globals.Tot_Week,
        name="bonmin",
        obj=1,
    )
