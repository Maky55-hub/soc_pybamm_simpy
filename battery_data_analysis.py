#!/usr/bin/env python
# coding: utf-8

# In[56]:


####################Implementing the Kalman Filter Method####################################
import pybamm
import numpy as np
import matplotlib.pyplot as plt

pybamm.set_logging_level('INFO')


# In[57]:


# Reverses the order of a list
# input: any list
# returns: the reversed-order list
def flipList(inputList):
    flippedList = list()
    for i in range(len(inputList)):
        flippedList.append(inputList[len(inputList) - i - 1])
    return flippedList


# Combines two lists to form a np.array
# inputs: two lists of x and y values
# returns: the np.array of the x and y values
def createArray(listX, listY):
    newList = list()
    for i in range(len(listX)):
        newList.append([listX[i], listY[i]])
    return np.array(newList)


# Creates a list of the modeled nth order polynomial fit values
# input: order n of the polynomial fit, list of experimental x values, list of experimental y values
# returns: a list of the modeled y-values using the polynomial fit
def getPolyFitValues(order, xList, yList):
    coefficients = np.polyfit(xList, yList, order)
    modeledValues = list()
    for x in xList:
        yVal = 0
        i = 0
        while order - i >= 0:
            yVal += coefficients[i] * x ** (order - i)
            i += 1
        modeledValues.append(yVal)
    return modeledValues


# Finds the chi-square value for experimental and modeled values
# inputs: list of experimental x-values, list of experimental y-values, list of modeled y-values
# returns: the chi-squared value
def getChiSquaredValue(experimentalValues, modeledValues):
    # Calculate the chi-squared value
    totalChiSquared = 0
    for i in range(len(experimentalValues)):
        expectedValue = modeledValues[i]
        observedValue = experimentalValues[i]
        if expectedValue != 0:
            totalChiSquared += abs(((observedValue - expectedValue)**2) / expectedValue)
        elif abs(observedValue - expectedValue) <= 0.000001:
            totalChiSquared += 0
        else:
            totalChiSquared += (observedValue + expectedValue)**2
    return totalChiSquared


# Finds the optimal order n of a polynomial fit using minimum chi-square analysis
# inputs: a list of experimental x values (OCV measurements) and list of experimental y values (normalized SoC)
# returns: a list object of [order n, chi-squared value of the nth order fit], where n is the optimal order fit
def findOptimalOrderFit(xValues, yValues):
    # Only checks order n=1:8 to minimize compute time
    n = 1
    chiSquaredResults = list()
    while n <= 8:
        currentChiSquared = getChiSquaredValue(yValues, getPolyFitValues(n, xValues, yValues))
        chiSquaredResults.append([n, currentChiSquared])
        n += 1
    # find the minimum order n
    minIndex = 0
    minChiSquared = 1000000000.0
    for i in range(len(chiSquaredResults)):
        if chiSquaredResults[i][1] < minChiSquared:
            minChiSquared = chiSquaredResults[i][1]
            minIndex = i
    # the returned object is of the form [order n, chi-squared value]
    return chiSquaredResults[minIndex]


# Prints the optimal order n, chi-squared value of the fit, and coefficients of the polynomial fit
# inputs: the chiSquaredResults output of findOptimalOrderFit(), a list of experimental x values (OCV measurements),
#         and list of experimental y values (normalized SoC)
# returns: none
def printFittingResults(chiSquaredResults, xValues, yValues):
    # prints the optimal order n
    print("Optimal Order Fit:", chiSquaredResults[0])
    # prints the chi-squared value of the fit
    print("Chi-Squared Value:", chiSquaredResults[1])
    # prints the coefficients of the optimal fit
    print("Coefficients:")
    coefficientList = np.polyfit(xValues, yValues, chiSquaredResults[0])
    order = chiSquaredResults[0]
    for i in range(len(coefficientList)):
        print("\tx^" + str(order - i) +":", coefficientList[i])


# In[62]:


class BatteryPolynomialFit:
    
    def __init__(self, battery_model, parameter_values):
        self.model = battery_model
        self.parameter_values = parameter_values
        self.Vmax = self.parameter_values['Upper voltage cut-off [V]']
        self.Vmin = self.parameter_values['Lower voltage cut-off [V]']
        self._voltage_reading_final = None
        self._normalized_SoC = None
        self._modeled_values = None
        self._coefficients_list = None
        self._main_solution = None
    
        self.calculate_coefficients()
    
    def calculate_coefficients(self):
        experiment = pybamm.Experiment([(
            f'Discharge at 0.2 C until {self.Vmin} V',
            'Rest for 4 hours',
            f'Charge at 0.2 C until {self.Vmax} V',
            f'Discharge at C/20 until {self.Vmin} V',
            f'Charge at 0.2 C until {self.Vmin + ((self.Vmax - self.Vmin) / 2)} V'
        )])

        simulation = pybamm.Simulation(
            model=self.model, 
            experiment=experiment, 
            parameter_values=self.parameter_values
        )
        self._main_solution = simulation.solve()
        sol = self._main_solution
        
        
        # Get the discharge step
        discharge_step = sol.cycles[0].steps[3]
        
        # Get the values of discharge capacity and OCV for the discharge step 
        measurements_adjusted = discharge_step['Discharge capacity [A.h]'].entries
        voltage_reading_final = flipList(discharge_step['Measured open circuit voltage [V]'].entries)

        num_measurements = len(measurements_adjusted)
        increment = 100.0/num_measurements
        normalized_SoC = list()
        tempI = 0
        while tempI < num_measurements:
            normalized_SoC.append(tempI*increment)
            tempI += 1


        # Finds the optimal order n of a polynomial fit of the data using minimum chi-square analysis
        optimal_fit_values = findOptimalOrderFit(voltage_reading_final, normalized_SoC)
        optimal_order = optimal_fit_values[0]

        # Creates a list of modeled SoC values using the optimal polynomial fit, used for plotting
        modeled_values = getPolyFitValues(optimal_order, voltage_reading_final, normalized_SoC)
        
        print("\nFITTING RESULTS")
        print("---------------")
        printFittingResults(optimal_fit_values, voltage_reading_final, normalized_SoC)
        
        # Calculate the coefficient list
        coefficients_list = np.polyfit(voltage_reading_final, normalized_SoC, optimal_fit_values[0])
        
        self._coefficients_list = coefficients_list
        self._modeled_values = modeled_values
        self._normalized_SoC = normalized_SoC
        self._voltage_reading_final = voltage_reading_final
        
    
    def plot_polynomial_fit(self):
        plt.plot(self.voltage_reading_final, self.normalized_SoC, label='Experimental Data')
        # Plots the modeled values using the polynomial fit for SoC vs OCV
        plt.plot(self.voltage_reading_final, self.modeled_values, label='Polynomial Fit')
        plt.legend(loc='best')
        plt.ylabel('State of Charge (%)')
        plt.xlabel('Cell Voltage (V)')
        plt.title("Battery State of Charge (SoC) vs\nOpen Circuit Voltage (OCV)", fontweight='bold')
        plt.grid(True)
        # Adjust spacing of subplots
        plt.subplots_adjust(wspace=0.35)
        plt.show()
        
    @property
    def last_solution(self):
        return self._main_solution.cycles[0].steps[4]
    
    @property
    def coefficients_list(self):
        return self._coefficients_list
    
    @property
    def modeled_values(self):
        return self._modeled_values
    
    @property
    def normalized_SoC(self):
        return self._normalized_SoC
    
    @property
    def voltage_reading_final(self):
        return self._voltage_reading_final

