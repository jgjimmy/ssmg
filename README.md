# ssmg
State-Space Ising Model code

This code uses the Multi-Graph State-Space Ising Model to create the figures shown in:
Gaudreault, J., Shimazaki, H. and Saxena, A. (2019). Online Estimation of Multiple Dynamic Graphs in Pattern Sequences. Available at: https://arxiv.org/abs/1901.07298
To generate the figures in that paper, please follow the following steps:
Figure 2 
1. Run “figure2_data_fit.py”. This generates data from a generative J matrix with 2 graphs and fits a model online. The first graph is likely to generate the image of a cross, while the second one is likely to generate images of a “T”. The weights follow sine waves. This will save the fitted J and weights at every time step in a pickle file “Data_fig2” in the “Data” folder. The folder will be created if it doesn’t exist.
2. Run “figure2.py”. This will generate the figure and save it in the “Figure” folder. The folder will be created if it doesn’t exist.

Figure 3
1. Run “gen_spikes_figure3.py”. This will generate spikes for a model with random graphs with weights following sine waves with different frequency. The trajectory is repeated 200 times, to create 200 epochs of data. The spike array obtained is save in a pickle file called “spikes_fig3” in the “Data” folder.
2. Run “figure3_data_fit.py”. This fits the Multi-Graph State-Space Ising Model to the data previously generated with number of graphs given by the variable “D_fit”. A data file containing the fitted weights at the last epoch of the data and the fitted J is saved in a pickle file called “Data_fig3_D_fit=X”, where “X” is the value of the variable “D_fit”. The file is saved in the “Data” folder. To replicate the figure, “figure3_data_fit.py” should be ran 4 times, with the value of “D_fit” manually set to 1, 2, 3 and 4.
3. Run “figure3.py”. This will generate the figure and save it in the “Figure” folder. The folder will be created if it doesn’t exist.

Figure 4
1. Download the data from Timme, Marshall, Bennett, Ripp, Lautzenhiser and Beggs grom CRCNS.org: 
N. M. Timme, N. J. Marshall, N. Bennett, M. Ripp, E. Lautzenhiser, and J. M. Beggs, “Criticality maximizes complexity in neural tissue,” Frontiers in Physiology, vol. 7, p. 425, 2016.

N. M. Timme, N. J. Marshall, N. Bennett, M. Ripp, E. Lautzenhiser, and J. M. Beggs, “Spontaneous spiking activity of thousands of neurons in rat hippocampal dissociated cultures,” CRCNS.org, 2016. [Online]. Available: http://dx.doi.org/10.6080/K0PC308P

2. Replace the Matlab data files in the “Neural_Data” folder with those downloaded from CRCNS.org.

3. Run “extract_data_spikes.py”. This extracts the list of spike times contained in the Matlab data files and creates spike arrays with 10ms time bins for the 12 neurons with the highest firing rate. The result is saved in a pickle file in the “Data” folder. The folder will be created if it doesn’t exist.
4. Run “figure4_data_fig.py”. This fits the Multi-Graph State-Space Ising Model to the spike array previously generated with number of graphs given by the variable “D_fit”. A data file containing the fitted weights at every time step of the data and the fitted J is saved in a pickle file called “Data_fig4_D_fit=X”, where “X” is the value of the variable “D_fit”. The file is saved in the “Data” folder. To replicate the figure, “figure4_data_fit.py” should be ran 3 times, with the value of “D_fit” manually set to 1, 2 and 3. By default (and in the paper), data from the 28th culture on day 34 is used. This can be changed by altering the values of the variables “n_culture” and “n_DIV”.
5. Run “figure4.py”. This will generate the figure and save it in the “Figure” folder. The folder will be created if it doesn’t exist. If the culture and/or day number was changed, “n_culture” and “n_DIV” need to be changed correspondingly. 
