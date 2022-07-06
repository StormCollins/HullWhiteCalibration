import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import QuantLib as ql
import seaborn as sns
from collections import namedtuple

displacement = 0

SwaptionData = namedtuple('SwaptionData', 'swaption_tenor, swap_tenor, vol, vol_type')


def load_discount_curve(file_path: str, sheet_name: str, day_count_convention: ql.DayCounter) -> ql.DiscountCurve:
    """
    Creates a QuantLib discount curve using dates and associated discount factors in an Excel file.

    The data needs to be set up in Excel as follows with the top left most cell being A1.

    Dates      | DFs
    -------------------
    2021-09-30 | 1.0000
    2021-11-03 | 0.9999
    2022-08-01 | 0.9988
    etc.

    :param file_path: The path of the market data
    :type file_path: str
    :param sheet_name: The name of the sheet containing the discount curve
    :type sheet_name: str
    :param day_count_convention: The day count convention
    :type day_count_convention: ql.DayCounter
    :return: A quantlib discount curve.
    :rtype: ql.DiscountCurve
    """
    data_frame: pd.DataFrame = pd.read_excel(file_path, sheet_name)
    dates: list[ql.Date] = [ql.Date(d.day, d.month, d.year) for d in data_frame['Dates']]
    dfs: list[float] = data_frame['DFs']
    return ql.DiscountCurve(dates, dfs, day_count_convention)


def load_swaption_data(file_path, sheet_name) -> list[SwaptionData]:
    """
    Imports 4 lists containing the: swaption tenors, associated swap tenors, associated vols, and associated vol types.

    :param file_path: The path of the market data
    :type file_path: str
    :param sheet_name: The name of the sheet containing the swaption data
    :type sheet_name: str
    :return: A list of named tuples (containing SwaptionData).
    :rtype: list[SwaptionData]
    """
    data_frame: pd.DataFrame = pd.read_excel(file_path, sheet_name)
    swaption_tenors: list[int] = data_frame['Swaption Tenors']
    swap_tenors: list[int] = data_frame['Swap Tenors']
    vols: list[float] = data_frame['Vols']
    vol_types: list[str] = data_frame['Vol Types']

    output: list[SwaptionData] = list()
    for i in range(0, len(swaption_tenors)):
        output.append(SwaptionData(int(swaption_tenors[i]), int(swap_tenors[i]), float(vols[i]), vol_types[i]))
    return output


def create_swaption_helpers(
        swaption_data: list[SwaptionData],
        index: ql.Index,
        term_structure: ql.DiscountCurve,
        swaption_pricing_engine) -> list[ql.SwaptionHelper]:
    """
    Creates a list of swaption helpers for use in calibration of a stochastic short rate model.

    :param swaption_data: A list of SwaptionData (a named tuple) which contains data relevant to creating swaptions.
    :type swaption_data: list[SwaptionData]
    :param index: The rate index.
    :type index: ql.Index
    :param term_structure: The discount curve.
    :type term_structure: ql.DiscountCurve
    :param swaption_pricing_engine: The swaption pricing engine.
    :return: A list of swaption helpers for calibration of a stochastic short rate model.
    :rtype: list[ql.SwaptionHelper]
    """
    nominal = 1.0
    swaption_helpers: list[ql.SwaptionHelper] = \
        [ql.SwaptionHelper(ql.Period(swaption.swaption_tenor, ql.Years),
                           ql.Period(swaption.swap_tenor, ql.Years),
                           ql.QuoteHandle(ql.SimpleQuote(swaption.vol)),
                           index,
                           index.tenor(),
                           index.dayCounter(),
                           index.dayCounter(),
                           term_structure,
                           ql.BlackCalibrationHelper.RelativePriceError,
                           ql.nullDouble(),
                           nominal,
                           ql.ShiftedLognormal,
                           displacement) for swaption in swaption_data]
    for swaption_helper in swaption_helpers:
        swaption_helper.setPricingEngine(swaption_pricing_engine)
    return swaption_helpers


def hull_white_calibration_report(
        hull_white_model: ql.HullWhite,
        swaption_helpers: list[ql.SwaptionHelper]):
    """
    Generates a report, in the console, detailing the Hull-White calibration results.

    :param hull_white_model: The Hull-White model
    :type hull_white_model: ql.HullWhite
    :param swaption_helpers: The list of swaption helpers whose vols are used in the calibration
    :type swaption_helpers: list[ql.SwaptionHelper]
    :return: Nothing - it prints a report in the console.
    :rtype: None
    """
    print('-' * 82)
    print('\tHull-White Calibration\n')
    alpha, sigma = hull_white_model.params()
    print(f'\talpha = {alpha:.5f}, sigma = {sigma:.5f}\n')
    price_and_vol_report(swaption_helpers)


def gsr_calibration_report(
        gsr_model: ql.Gsr,
        vol_step_dates: list[ql.Date],
        swaption_helpers: list[ql.SwaptionHelper]):
    """
    Generates a report, in the console, detailing the Hull-White calibration results.

    :param gsr_model: The GSR (Gaussian Short Rate) model
    :type gsr_model: ql.Gsr
    :param vol_step_dates: The dates at which the vol changes.
    :type vol_step_dates: list[ql.Date]
    :param swaption_helpers:
    :return: Nothing - it prints a report in the console.
    :rtype: None
    """
    print('-' * 82)
    print('\tGSR Calibration\n')
    params = [x for x in gsr_model.params()]
    alpha = params[0]
    sigmas = params[1:]

    print(f'\talpha = {alpha:.5f}')
    print(f'\tDate\t\tSigma')
    for i in range(0, len(vol_step_dates)):
        print(f'\t{vol_step_dates[i].to_date()}\t{sigmas[i]:.5f}')
    print(' ')
    price_and_vol_report(swaption_helpers)


def price_and_vol_report(swaption_helpers: list[ql.SwaptionHelper]):
    """
    Generates a sub-report comparing the swaption implied and market vols and prices.

    :param swaption_helpers: The swaption helpers used in the stochastic short rate model calibration
    :type swaption_helpers: list[ql.SwaptionHelper]
    :return: Nothing - outputs results to a console.
    :rtype: None
    """
    print(f'{"Model Price":>15} {"Market Price":>15} {"Implied Vol":>15} {"Market Vol":>15} {"Rel Error":>15}')
    print('-' * 82)
    cum_err = 0.0
    for i, s in enumerate(swaption_helpers):
        model_price = s.modelValue()
        market_vol = swaption_data[i].vol
        black_price = s.blackPrice(market_vol)
        rel_error = model_price / black_price - 1.0
        implied_vol = s.impliedVolatility(model_price, 1e-5, 50, 0.0, 0.50)
        rel_error2 = implied_vol / market_vol - 1.0
        cum_err += rel_error2 * rel_error2
        print(f'{model_price:15.5f} {black_price:15.5f} {implied_vol:15.5f} {market_vol:15.5f} {rel_error:15.5f}')
    print('-' * 82)
    print(f'Cumulative Error: {math.sqrt(cum_err):<15.5f}')


def plot_process(axis, t, hull_white_process, alpha):
    sorted_hull_white_process = hull_white_process[:, np.argsort(np.average(hull_white_process, 0))]
    for j in range(0, hull_white_process.shape[1]):


today = ql.Date(1, ql.April, 2022)
settlement = ql.Date(30, ql.April, 2022)
ql.Settings.instance().evaluationDate = today
market_data_file: str = r'C:\GitLab\HullWhiteCalibration\market-data-example-2-coterminal-swaptions.xlsx'

discount_curve = \
    load_discount_curve(market_data_file, 'discount-curve', ql.Actual365Fixed())
term_structure = ql.YieldTermStructureHandle(discount_curve)

jibar = ql.Jibar(ql.Period('3M'), term_structure)

hull_white = ql.HullWhite(term_structure)
jamshidian_engine = ql.JamshidianSwaptionEngine(hull_white)
swaption_data: list[SwaptionData] = load_swaption_data(market_data_file, 'swaption-data')
swaption_helpers: list[ql.SwaptionHelper] = \
    create_swaption_helpers(swaption_data, jibar, term_structure, jamshidian_engine)
optimization_method = ql.LevenbergMarquardt(1.0e-8, 1.0e-8, 1.0e-8)
end_criteria = ql.EndCriteria(400, 100, 1e-8, 1e-8, 1e-8)
hull_white.calibrate(swaption_helpers, optimization_method, end_criteria)
hull_white_calibration_report(hull_white, swaption_helpers)

# It seems GSR requires a curve of at least 60 years to calibrate?!?!
# vol_step_dates = [today + 365 * x for x in [1, 2, 3, 4, 5, 7]]
# vols = [ql.QuoteHandle(ql.SimpleQuote(0.01)), ql.QuoteHandle(ql.SimpleQuote(0.01)),
#         ql.QuoteHandle(ql.SimpleQuote(0.01)), ql.QuoteHandle(ql.SimpleQuote(0.01)),
#         ql.QuoteHandle(ql.SimpleQuote(0.01)), ql.QuoteHandle(ql.SimpleQuote(0.01)),
#         ql.QuoteHandle(ql.SimpleQuote(0.01))]
# reversions = [ql.QuoteHandle(ql.SimpleQuote(0.01))]
# gsr = ql.Gsr(term_structure, vol_step_dates, vols, reversions)
# gaussian_swaption_engine = ql.Gaussian1dSwaptionEngine(gsr, 64, 7.0, True, False, term_structure)
# nonstandardSwaptionEngine = \
#     ql.Gaussian1dNonstandardSwaptionEngine(gsr, 64, 7.0, True, False, ql.QuoteHandle(ql.SimpleQuote(0)), term_structure)
# gsr_swaption_helpers = create_swaption_helpers(swaption_data, jibar, term_structure, gaussian_swaption_engine)
# gsr.calibrate(gsr_swaption_helpers, optimization_method, end_criteria)
# gsr_calibration_report(gsr, vol_step_dates, gsr_swaption_helpers)



time_grid = ql.TimeGrid(10, 100)
sequence_generator = ql.UniformRandomSequenceGenerator(len(time_grid), ql.UniformRandomGenerator())
gaussian_sequence_generator = ql.GaussianRandomSequenceGenerator(sequence_generator)
alpha = hull_white.params()[0]
sigma = hull_white.params()[1]
hull_white_process = ql.HullWhiteProcess(term_structure, alpha, sigma)
path_generator = ql.GaussianPathGenerator(hull_white_process, time_grid[-1], len(time_grid), gaussian_sequence_generator, False)
number_of_paths = 1000

paths = np.zeros(shape=(number_of_paths, len(time_grid)))

for i in range(number_of_paths):
    path = path_generator.next().value()
    paths[i, :] = np.array([path[j] for j in range(len(time_grid))])

time = np.linspace(0, 10, 101)
plt.plot(time, paths.transpose())
plt.show()
