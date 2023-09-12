"""
Image based video search engine: prototype
This file contains the functions that were used to create evaluation plots and evaluate the HPO data
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
import joblib

from main import method_selector


def main():
    """
    main body to call plotting functions, uncomment when neccesary
    :return:
    """
    # plot_methods_total_time()
    #
    # plot_data_timevsk_diff_frames()
    #
    # store_hpo_lsh_data(r'.\test_data\hpo_results\faiss_lsh270.pkl', 270)
    # store_hpo_lsh_data(r'.\test_data\hpo_results\faiss_lsh8100.pkl', 8100)
    # store_hpo_lsh_data(r'.\test_data\hpo_results\faiss_lsh50000.pkl', 50000)
    #
    # store_hpo_ivf_data(r'.\test_data\hpo_results\faiss_ivf270.pkl')
    # store_hpo_ivf_data(r'.\test_data\hpo_results\faiss_ivf8100.pkl')
    #
    # store_hpo_hnsw_data(r'.\test_data\hpo_results\faiss_hnsw270.pkl')

    get_final_intersection_points()

    # plot_methods_total_time_final()
    # break_points_15min()
    # plot_selector()
    #
    # check_recall_precision_stored_data()
    # validation_random()


def plot_data_timevsk_diff_frames():
    """
    Plot k_percentage against time,recall and precision for 3 different amounts of frames.
    :return: 3 plots containing the results
    """
    filename270 = r'test_data/timevsk270.csv'
    filename8100 = r'test_data/timevsk8100.csv'
    filename50000 = r'test_data/timevsk50000.csv'
    df270 = pd.read_csv(os.path.abspath(filename270))
    df8100 = pd.read_csv(os.path.abspath(filename8100))
    df50000 = pd.read_csv(os.path.abspath(filename50000))

    df_methods270 = []
    df_methods8100 = []
    df_methods50000 = []

    methods = ['linear', 'faiss_flat_cpu', 'faiss_flat_gpu', 'faiss_hnsw', 'faiss_ivf_cpu', 'faiss_pq', 'faiss_lsh']
    linestyles = ['black -', 'magenta -', 'cyan -', 'black --', 'red --', 'blue --', 'magenta --']

    for method in methods:
        df_methods270.append(df270.query(f'method == "{method}"'))
        df_methods8100.append(df8100.query(f'method == "{method}"'))
        df_methods50000.append(df50000.query(f'method == "{method}"'))

    k_percentage = np.linspace(0.1, 25, 50)

    fig1, axes1 = plt.subplots(1, 3, figsize=(12, 6))
    plt.subplots_adjust(right=0.85, wspace=0.3)
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 6))
    plt.subplots_adjust(right=0.85, wspace=0.3)
    fig3, axes3 = plt.subplots(1, 3, figsize=(12, 6))
    plt.subplots_adjust(right=0.85, wspace=0.3)

    for linestyle, df_method270, df_method8100, df_method50000 in zip(linestyles, df_methods270, df_methods8100,
                                                                      df_methods50000):
        color, style = linestyle.split()
        print(color, style)
        axes1[0].plot(k_percentage, df_method270['searchtime'], style, color=color)
        axes1[1].plot(k_percentage, df_method270['mAP'], style, color=color)
        axes1[2].plot(k_percentage, df_method270['recall'], style, color=color)

        axes2[0].plot(k_percentage, df_method8100['searchtime'], style, color=color)
        axes2[1].plot(k_percentage, df_method8100['mAP'], style, color=color)
        axes2[2].plot(k_percentage, df_method8100['recall'], style, color=color)

        axes3[0].plot(k_percentage, df_method50000['searchtime'], style, color=color)
        axes3[1].plot(k_percentage, df_method50000['mAP'], style, color=color)
        axes3[2].plot(k_percentage, df_method50000['recall'], style, color=color)

    axes1_new = [axes1[0], axes2[0], axes3[0]]
    axes2_new = [axes1[1], axes2[1], axes3[1]]
    axes3_new = [axes1[2], axes2[2], axes3[2]]

    k_percent = 7

    for ax1, ax2, ax3 in zip(axes1_new, axes2_new, axes3_new):
        ax1.title.set_text("search time")
        ax1.set(xlabel="K (%)", ylabel="Search time (s)")
        ax1.set_xlim([0, 25])
        ax1.grid(True)

        ax2.title.set_text("mAP")
        ax2.set(xlabel="K (%)", ylabel="mAP")
        ax2.set_xlim([0, 25])
        ax2.set_ylim([0.2, 0.85])
        ax2.grid(True)
        ax2.axvline(x=k_percent, color='black', linestyle='--', linewidth=0.75)
        ax2.axhline(y=0.65, color='black', linestyle='--', linewidth=0.75)

        ax3.title.set_text("recall")
        ax3.set(xlabel="K (%)", ylabel="recall")
        ax3.set_xlim([0, 25])
        ax3.set_ylim([0, 0.95])
        ax3.axvline(x=k_percent, color='black', linestyle='--', linewidth=0.75)
        ax3.axhline(y=0.50, color='black', linestyle='--', linewidth=0.75)
        ax3.grid(True)

    methods_legend = ['L2', 'FAISS flat L2 CPU', 'FAISS flat L2 GPU',
                      'FAISS HNSW', 'FAISS IVF CPU', 'FAISS PQ', "FAISS LSH"]
    fig1.legend(methods_legend, loc="center right", title="Method", borderaxespad=0.1)
    fig2.legend(methods_legend, loc="center right", title="Method", borderaxespad=0.1)
    fig3.legend(methods_legend, loc="center right", title="Method", borderaxespad=0.1)

    fig1.savefig("test_data/plots/timevskfinal270.png")
    fig2.savefig("test_data/plots/timevskfinal8100.png")
    fig3.savefig("test_data/plots/timevskfinal50000.png")

    plt.show()


def plot_methods_total_time():
    """
    Plot for the methods the time per query for 3 different amounts of frames
    :return: 3 plots with the time times needed for each method
    """
    methods = ['L2', 'ANNOY', 'HNSW', 'HNSW batch', 'FAISS flat L2 CPU', 'FAISS flat L2 GPU',
               'FAISS HNSW', 'FAISS IVF CPU', 'FAISS IVF GPU', "FAISS PQ", "FAISS LSH"]
    max_queries = 1000
    n_queries = [1, max_queries]

    # Manually retrieved data for n_frames 270
    search_times_270 = np.array([0.98, 0.5, 0.09, 0.08, 0.33, 0, 0.01, 0, 0.02, 0, 0.02])
    search_times_max_270 = search_times_270 * max_queries
    build_times_270 = np.array([0, 48.04, 28.19, 21.02, 12.18, 453.41, 9.01, 22.02, 468.68, 70.06, 53.05])
    total_270 = search_times_270 + build_times_270
    total_max_270 = search_times_max_270 + build_times_270

    # Manually retrieved data for n_frames 8100
    search_times_8100 = np.array([43.59, 3.91, 0.76, 0.75, 0.3, 0.3, 0.25, 0.15, 0.13, 0.04, 0.04])
    search_times_max_8100 = search_times_8100 * max_queries
    build_times_8100 = np.array([0, 1038.43, 1192.34, 1118.12, 13.01, 516.25, 237.22, 105.09, 594.22, 1120.82, 81.07])
    total_8100 = search_times_8100 + build_times_8100
    total_max_8100 = search_times_max_8100 + build_times_8100

    # Manually retrieved data for n_frames 50000
    search_times_50000 = np.array([282.33, 18.6, 5.52, 5.54, 1.57, 0.1, 19.33, 1.48, 0.39, 0.31, 0.24])
    search_times_max_50000 = search_times_50000 * max_queries
    build_times_50000 = np.array([0, 6885.22, 9514.22, 4036.03, 84.08, 558.6, 4414.98, 495.54, 857.87, 6999.24, 291.26])
    total_50000 = search_times_50000 + build_times_50000
    total_max_50000 = search_times_max_50000 + build_times_50000

    # styles = ['blue -','green --','red -.','cyan :','magenta -','yellow --','black -.','forestgreen :','blueviolet -','orangered --','hotpink -.']
    styles = ['black -', 'red -', 'green -', 'blue -', 'magenta -', 'cyan -',
              'black --', 'red --', 'green --', 'blue --', 'magenta --', 'cyan --']
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))

    for style, time_270, time_max_270, time_8100, time_max_8100, time_50000, time_max_50000 \
            in zip(styles, total_270, total_max_270, total_8100, total_max_8100, total_50000, total_max_50000):
        (color, linestyle) = style.split()  # Split the color and linestyle
        ax1.plot(n_queries, [time_270, time_max_270], linestyle, color=color)
        ax2.plot(n_queries, [time_8100, time_max_8100], linestyle, color=color)
        ax3.plot(n_queries, [time_50000, time_max_50000], linestyle, color=color)

    fig.suptitle("Comparison of total time for each method")
    fig.legend([f"{methods[0]}", f"{methods[1]}", f"{methods[2]}",
                f"{methods[3]}", f"{methods[4]}", f"{methods[5]}",
                f"{methods[6]}", f"{methods[7]}", f"{methods[8]}",
                f"{methods[9]}", f"{methods[10]}"], loc="center right", title="Method", borderaxespad=0.1)
    plt.subplots_adjust(right=0.75, hspace=0.8)

    ax1.title.set_text("(a) 270 keyframes")
    ax1.set_ylim([0, 60])
    ax1.set_xlim([0, max_queries])
    ax1.set(xlabel="number of queries ", ylabel="Time (ms)")
    ax1.grid(True)

    ax2.title.set_text("(b) 8100 keyframes")
    ax2.set_ylim([0, 1000])
    ax2.set_xlim([0, max_queries])
    ax2.set(xlabel="number of queries ", ylabel="Time (ms)")
    ax2.grid(True)

    ax3.title.set_text("(c) 50000 keyframes")
    ax3.set_ylim([0, 1250])
    ax3.set_xlim([0, max_queries])
    ax3.set(xlabel="number of queries ", ylabel="Time (ms)")
    ax3.grid(True)

    plt.savefig("test_data/plots/total_time_per_method.png")
    plt.show()


def plot_methods_total_time_final():
    """
    Plot for the selected and optimized methods the time per query for 3 different amounts of frames
    :return: 3 plots with the time times needed for each method
    """
    methods = ['L2', 'FAISS flat L2 CPU', 'FAISS flat L2 GPU',
               'FAISS HNSW', 'FAISS IVF CPU', "FAISS LSH"]
    max_queries = 1000
    n_queries = [1, max_queries]

    # Manually retrieved data for n_frames 270
    search_times_270 = np.array([0.75, 0.16, 0.0, 0.0, -1, 0.01])
    search_times_max_270 = search_times_270 * max_queries
    build_times_270 = np.array([0, 1.0, 405.0, 12.01, -1, 8.01])
    total_270 = search_times_270 + build_times_270
    total_max_270 = search_times_max_270 + build_times_270

    # Manually retrieved data for n_frames 8100
    search_times_8100 = np.array([33.65, 0.35, 0.02, 0.11, -1, 0.03])
    search_times_max_8100 = search_times_8100 * max_queries
    build_times_8100 = np.array([0, 9.01, 414.89, 206.19, -1, 39.03])
    total_8100 = search_times_8100 + build_times_8100
    total_max_8100 = search_times_max_8100 + build_times_8100

    # Manually retrieved data for n_frames 50000
    search_times_50000 = np.array([219.54, 1.31, 0.09, 3.43, -1, 0.2])
    search_times_max_50000 = search_times_50000 * max_queries
    build_times_50000 = np.array([0, 54.96, 466.42, 5045.29, -1, 131.12])
    total_50000 = search_times_50000 + build_times_50000
    total_max_50000 = search_times_max_50000 + build_times_50000

    ivf_n_queries270 = np.array([1, 123, 1000])
    ivf_n_queries8100 = np.array([1, 5, 11, 90, 259, 509, 1000])
    ivf_n_queries50000 = n_queries

    ivf_search_times270 = np.array([0.0, 0.01, 0.01])
    ivf_search_times8100 = np.array([2, 0.2, 0.27, 0.17, 0.14, 0.11, 0.11])
    ivf_search_times50000 = np.array([0.96, 0.96])

    ivf_build_time270 = np.array([11.01, 12.01, 12.01])
    ivf_build_time8100 = np.array([83.07, 84.08, 81.18, 89.08, 90.08, 112.35, 112.35])
    ivf_build_time50000 = np.array([452.41, 452.41])

    ivf_total_270 = ivf_build_time270 + (ivf_n_queries270 * ivf_search_times270)
    ivf_total_8100 = ivf_build_time8100 + (ivf_n_queries8100 * ivf_search_times8100)
    ivf_total_50000 = ivf_build_time50000 + (ivf_n_queries50000 * ivf_search_times50000)

    styles = ['black -', 'magenta -', 'cyan -', 'black --', 'red --', 'magenta --']
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))

    for style, time_270, time_max_270, time_8100, time_max_8100, time_50000, time_max_50000 \
            in zip(styles, total_270, total_max_270, total_8100, total_max_8100, total_50000, total_max_50000):
        (color, linestyle) = style.split()  # Split the color and linestyle
        if color == "red":
            ax1.plot(ivf_n_queries270, ivf_total_270, linestyle, color=color)
            ax2.plot(ivf_n_queries8100, ivf_total_8100, linestyle, color=color)
            ax3.plot(ivf_n_queries50000, ivf_total_50000, linestyle, color=color)

        else:
            ax1.plot(n_queries, [time_270, time_max_270], linestyle, color=color)
            ax2.plot(n_queries, [time_8100, time_max_8100], linestyle, color=color)
            ax3.plot(n_queries, [time_50000, time_max_50000], linestyle, color=color)

    fig.suptitle("Comparison of total time for each method")
    fig.legend([f"{methods[0]}", f"{methods[1]}", f"{methods[2]}",
                f"{methods[3]}", f"{methods[4]}", f"{methods[5]}"], loc="center right", title="Method",
               borderaxespad=0.1)
    plt.subplots_adjust(right=0.75, hspace=0.8)

    ax1.title.set_text("(a) 270 keyframes")
    ax1.set_ylim([0, 30])
    ax1.set_xlim([0, max_queries])
    ax1.set(xlabel="number of queries ", ylabel="Time (ms)")
    ax1.grid(True)

    ax2.title.set_text("(b) 8100 keyframes")
    ax2.set_ylim([0, 500])
    ax2.set_xlim([0, max_queries])
    ax2.set(xlabel="number of queries ", ylabel="Time (ms)")
    ax2.grid(True)

    ax3.title.set_text("(c) 50000 keyframes")
    ax3.set_ylim([0, 750])
    ax3.set_xlim([0, max_queries])
    ax3.set(xlabel="number of queries ", ylabel="Time (ms)")
    ax3.grid(True)

    plt.savefig("test_data/plots/total_time_per_method_final.png")
    plt.show()


def store_hpo_lsh_data(filepath, number):
    """
    Store the data from the HPO for LSH in a latex table and create plots
    :param filepath: The file in which the optuna study is stored
    :param number: The number of frames the HPO was done for
    :return: A latex file with a table and plots for the HPO
    """
    n_queries = 1000
    query_range = [1, n_queries]

    # Load the study and create a dataframe
    object_study = joblib.load(filepath)
    df = object_study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))

    # Create a latex and png file name
    latexfile = filepath.split("\\")[-1].split(".")[0] + '.tex'
    plotfile = 'hpo_' + filepath.split("\\")[-1].split(".")[0] + '.png'
    # Store the study data as a latex table
    with open(rf'.\test_data\latex\{latexfile}', 'w') as savefile:
        savefile.write(df.to_latex(position='h!', label='app:rawFaissLSH', index=False,
                                   caption=f'Filler for caption'))

    # Filter the study data based on requirements for mAP and recall
    selection = df.query("values_0 >= 0.65 & values_1 >= 0.5 & state == 'COMPLETE'").sort_values(
        by="params_bitlength_percentage")

    # Linear interpolate data from 1 to 1000 queries
    search_times = selection['values_2']
    search_times1000 = search_times * n_queries
    build_times = selection['values_3']
    total_times = build_times + search_times
    total_times1000 = build_times + search_times1000

    # plot the data
    plt.rcParams["figure.figsize"] = (15, 8)
    linestyles = ['-', '--', '-.']
    for index, (time, time1000) in enumerate(zip(total_times, total_times1000)):
        plt.plot(query_range, [time, time1000], linestyles[index // 10])

    plt.xlim([0, n_queries])
    plt.gcf().legend(selection['params_bitlength_percentage'].round(2), loc="center right",
                     title="bitlength percentage", borderaxespad=0.1)
    plt.subplots_adjust(right=0.85)
    plt.xlabel("number of queries")
    plt.ylabel("total time (s)")
    plt.title(f"Paramater optimization for FAISS LSH at {number} keyframes.")
    plt.savefig(fr".\test_data\plots\{plotfile}")
    plt.show()


def store_hpo_ivf_data(filepath):
    """
    Store the data from the HPO for IVF in a latex table and create plots
    :param filepath: The file in which the optuna study is stored
    :return: A latex file with a table for the HPO
    """
    n_queries = 1000

    # Load the study and create a dataframe
    object_study = joblib.load(filepath)
    df = object_study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    # Create a latex file name and store the study data as a latex table
    latexfile = filepath.split("\\")[-1].split(".")[0] + '.tex'
    with open(rf'.\test_data\latex\{latexfile}', 'w') as savefile:
        savefile.write(df.to_latex(position='h!', label='app:rawFaissLSH', index=False,
                                   caption=f'Filler for caption'))

    # Filter the data based on the requirements
    selection = df.query("values_0 >= 0.65 & values_1 >= 0.5 & state == 'COMPLETE'")
    search_times = selection['values_2']
    build_times = selection['values_3']

    # Calculate the breakpoints of the collected data
    minima, breakpoints = calc_breakpoints(build_times, search_times, n_queries, len(selection))

    breakpoints = np.array(breakpoints)
    minima = np.array(minima)
    print(f"Breakpoints at indices: {breakpoints}")
    print(f"Dataframe entries of the indices: {minima[breakpoints]}")


def store_hpo_hnsw_data(filepath):
    """
    Store the data from the HPO for HNSW in a latex table and create plots
    :param filepath: The file in which the optuna study is stored
    :return: A latex file with a table for the HPO
    """
    n_queries = 1000
    # Load the study and create a dataframe
    object_study = joblib.load(filepath)
    df = object_study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    # Create a latex file name and store the study data as a latex table
    latexfile = filepath.split("\\")[-1].split(".")[0] + '.tex'
    with open(rf'.\test_data\latex\{latexfile}', 'w') as savefile:
        savefile.write(df.to_latex(position='h!', label='app:rawFaissLSH', index=False,
                                   caption=f'Filler for caption'))

    # Filter the data based on the requirements
    selection = df.query("values_0 >= 0.65 & values_1 >= 0.5 & state == 'COMPLETE'")
    search_times = selection['values_2']
    build_times = selection['values_3']

    # Calculate the breakpoints of the collected data
    minima, breakpoints = calc_breakpoints(search_times, build_times, n_queries, len(selection))

    breakpoints = np.array(breakpoints)
    minima = np.array(minima)
    print(f"Breakpoints at indices: {breakpoints}")
    print(f"Dataframe entries of the indices: {minima[breakpoints]}")


def calc_breakpoints(build_times, search_times, n_queries, n_entries):
    """
    Calculate breakpoints between fastest methods
    :param build_times: The array of build times per thing to be evaluared (HPO run/method/etc.)
    :param search_times: TThe array of search times per thing to be evaluared (HPO run/method/etc.)
    :param n_queries: The amount of queries to be evaluated
    :param n_entries: The amount of entries to be evaluated (HPO runs/methods/etc.)
    :return: The minima and breakpoints of the data
    """
    # Find the minima for each query
    minima = []
    for i in range(n_queries):
        minimum = np.inf
        min_ind = None
        for j in range(n_entries):
            res = build_times[j] + search_times[j] * i
            if res < minimum:
                minimum = res
                min_ind = j
        minima.append(min_ind)

    # Find the breakpoints corresponding to the minima
    breakvalue = minima[0]
    breakpoints = [0]
    for ind, value in enumerate(minima):
        if value != breakvalue:
            breakvalue = value
            breakpoints.append(ind)
    return np.array(minima), np.array(breakpoints)


def get_final_intersection_points():
    """
    Get the intersection points of the final used methods
    :return: the intersection points and which method is used
    """
    methods = ['L2', 'FAISS flat L2 CPU', 'FAISS flat L2 GPU',
               'FAISS HNSW', "FAISS LSH", 'FAISS IVF CPU', ]
    methods = np.array(methods)

    # Manually retrieved data for n_frames 270
    search_times_270 = np.array([0.75, 0.16, 0.0, 0.0, 0.01])
    build_times_270 = np.array([0, 1.0, 405.0, 12.01, 8.01])

    # Manually retrieved data for n_frames 8100
    search_times_8100 = np.array([33.65, 0.35, 0.02, 0.11, 0.03])
    build_times_8100 = np.array([0, 9.01, 414.89, 206.19, 39.03])

    # Manually retrieved data for n_frames 50000
    search_times_50000 = np.array([219.54, 1.31, 0.09, 3.43, 0.2])
    build_times_50000 = np.array([0, 54.96, 466.42, 5045.29, 131.12])

    build_times_per_frames = [build_times_270, build_times_8100, build_times_50000]
    search_times_per_frames = [search_times_270, search_times_8100, search_times_50000]

    minima_per_frames = []
    for ind, n_frames in enumerate([270, 8100, 50000]):
        minima = []
        for i in range(1000):
            minimum = np.inf
            min_ind = None
            for j in range(len(methods) - 1):
                res = build_times_per_frames[ind][j] + search_times_per_frames[ind][j] * i
                if res < minimum:
                    minimum = res
                    min_ind = j
            minima.append(min_ind)
        minima_per_frames.append(minima)

    for minima_per_frame in minima_per_frames:
        breakvalue = minima_per_frame[0]
        breakpoints = [0]
        for ind, value in enumerate(minima_per_frame):
            if value != breakvalue:
                breakvalue = value
                breakpoints.append(ind)

        minima_per_frame = np.array(minima_per_frame)
        breakpoints = np.array(breakpoints)
        print(f"Methods: {methods[minima_per_frame[breakpoints]]}")
        print(f"Breakpoints:{breakpoints}")


def break_points_15min():
    """
    Determines the breakpoints for the selector
    :return: A file that contains the breakpoint data for the selector
    """
    # Load in the data and specify the same number of frames used
    df = pd.read_csv(r".\test_data\interp_selector_data_2.csv")
    data = np.array([np.arange(270, 4050, 270)])
    data = np.append(data, np.array([np.arange(4050, 50000, 4050)]))
    data = np.append(data, 50000)
    all_data = []
    # Make selections of data based on the amount of frames
    for i in data:
        print(f"n_frames = {i}/50000")
        df_selec = df.query(f"n_frames=={i}")
        search_time = df_selec.iloc[:, 3].to_numpy()
        build_time = df_selec.iloc[:, 2].to_numpy()

        method_array, breaks = calc_breakpoints(build_time, search_time, 1000, 5)
        method_array[0] = method_array[1]
        all_data.append(method_array)

    # Make one big list of all the data and save it
    all_data = np.concatenate(all_data).ravel()
    filename = r".\test_data\interp_data.npy"
    np.save(filename, np.array(all_data))


def check_recall_precision_stored_data():
    """
    check recall and precision of data stored of manual runs
    :return:
    """
    df = pd.read_csv(r".\test_data\15minresults.csv")
    selection = df[df.iloc[:, 4] <= 0.65]
    selection2 = df[df.iloc[:, 5] < 0.5]
    print(selection)
    print(selection2)


def validation_random():
    """
    check if the selected method of manual runs was correct
    :return:
    """
    df = pd.read_csv(r".\test_data\15minresults6.csv")
    df.info()
    indices = []
    for i in range(0, len(df), 6):
        selec = df.loc[i:i + 5]
        n_queries = df.loc[i].iloc[1]
        minima = selec.iloc[:, 3] + selec.iloc[:, 4] * n_queries
        indices.append(minima.idxmin())
        print(min(minima))
    print(df.loc[indices])


def plot_selector():
    """
    Make a visualization of the selector
    :return: a plot that visualizes the selector
    """
    methods = ["linear", "faiss_flat_cpu", "faiss_flat_gpu", "faiss_hnsw", "faiss_lsh", "faiss_ivf"]
    steps = 100
    total = []
    for i in range(1, 50002, int(50000 / steps)):
        data = []
        for j in range(1, 1002, int(1000 / steps)):
            data.append(method_selector(i, j, use_indices=True))
            print(f"{i, j}:/50000,1000")
        total.append(data)

    x, y = np.meshgrid(range(1, 1002, int(1000 / steps)), range(1, 50002, int(50000 / steps)))

    cmap = mpl.cm.Set1
    norm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)

    plt.pcolormesh(x, y, total, cmap=cmap, clim=(0, 6))
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=[0, 1, 2, 3, 4, 5])
    cbar.set_ticklabels(methods)
    plt.title("Interpolation of method selector")
    plt.ylabel("number of keyframes")
    plt.xlabel("number of queries")
    plt.xlim([0, 1000])
    plt.ylim([0, 50000])
    plt.savefig(os.path.abspath(r".\test_data\method_selectoraq2q1.png"))
    plt.show()


if __name__ == "__main__":
    main()
