import numpy as np
import pandas as pd
import random
import math

class DataGenerator:
    all_functions = [
        (lambda x, sign, c, n, i: sign * c * x),
        (lambda x, sign, c, n, i: sign * ((c * x) ** n)),
        (lambda x, sign, c, n, i: sign * ((c * x) ** (1/n))),
        (lambda x, sign, c, n, i: sign * math.log(abs(c) * x + 1, n + 1)),
        (lambda x, sign, c, n, i: sign * (x and i)),
        (lambda x, sign, c, n, i: sign * (x or i)),
        (lambda x, sign, c, n, i: sign * (x ^ i))
        ]

    def __init__(self, num_cols, num_rows=10, num_important=1, num_interaction_terms=None, interaction_type='all', effects=None, frequencies={}, correlation_range=[-0.9, 0.9], target='target', intercept=0, noise=0):
        # Record initialization params
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.num_important = num_important
        self.num_interaction_terms = num_interaction_terms if num_interaction_terms is not None else self.num_important
        self.interaction_type = interaction_type
        self.effects = effects
        self.frequencies = frequencies
        self.correlation_range = correlation_range
        self.target = target
        self.intercept = intercept
        self.noise = noise

        # Generate default parameters
        self.rng = np.random.default_rng()
        self.cols = range(num_cols)
        self.important_variables = self.cols[:num_important]
        self.interaction_terms = self.cols[-self.num_interaction_terms:]

        self.importances = [1 if var in self.important_variables else 0 for var in self.cols]

        for col in self.cols:
            if col not in frequencies.keys():
                frequencies[col] = self.rng.random()

        # Choose functions according to interaction type
        self.functions = DataGenerator.all_functions
        
        if interaction_type == 'all':
                self.functions = DataGenerator.all_functions
        elif interaction_type =='linear':
                self.functions = DataGenerator.all_functions[0:1]

        self.interactions = self.generate_interactions()

        # Check if effects is a string, if so change target functions
        self.target_functions = DataGenerator.all_functions

        if effects == 'all':
            self.target_functions = DataGenerator.all_functions
        elif effects == 'linear':
            self.target_functions = DataGenerator.all_functions[0:1]
        elif effects == 'constant':
            self.effects = [(lambda x: x) if i in self.important_variables else (lambda x: 0) for i in self.cols]
        
        # If effects wasn't a listlike structure, generate effects according to specifications
        if self.effects is None or type(self.effects) == str:
            self.effects = self.random_interaction(self.important_variables, functions=self.target_functions)
    
    def random_interaction(self, interacting_variables, cols=None, functions=None):
        """
        Generates a pandas Series of lambda functions representing diverse interaction terms for binary data.
        Each function corresponds to a column in 'cols'.

        Note that this function is meant to provide a list of functions that will all be applied and then summed
        in order to get the value of a single column in the generated data. Use the generate_interactions
        function to generate all interactions for a generated dataset.

        Parameters:
        - cols (list or Series): List of all column indices in the dataset
        - interacting_variables (list or Series): List of column indices within 'cols' for which to
          generate specific interaction terms based on random selections of interactions.
        - functions (list or Series of functions): List of functions to choose from when generating interactions.
          Should have parameters n, c, i, and sign

        Returns:
        - series of lambda functions: Each function is designed to apply a specific interaction
          to its input, based on the type of interaction randomly assigned to its corresponding column.
        """
        # Set class values if parameters None
        cols = cols if cols is not None else self.cols
        functions = functions if functions is not None else self.functions

        interaction_list = [lambda x: 0 for col in cols]

        for col in interacting_variables:
            # Generate random values
            roll = random.choice(range(len(functions)))
            n = random.randint(1, 10)
            c = random.randint(1, 10)
            i = random.randint(0, 1)
            sign = random.choice([-1, 1])

            f = lambda x, roll=roll, n=n, c=c, i=i, sign=sign: functions[roll](x, sign=sign, c=c, n=n, i=i)

            interaction_list[col] = f

        return pd.Series(interaction_list)

    def generate_interactions(self, cols=None, interaction_terms=None, important_variables=None, correlation_range=None):
        """
        Generates interaction terms for a dataset by selecting random samples of columns and creating interaction
        functions for them. This function orchestrates the creation of a comprehensive dataframe of interaction
        terms, combining both targeted columns and a subset of other columns to enrich the dataset's features with
        interactions. Apply the generated interaction functions across a dataset for generated interaction terms

        This function relies on 'random_interaction' to create specific interaction functions for each term and 'random_interactions'
        to compile these into a dictionary format suitable for application across a dataset.

        Parameters:
        - cols (list or Series): List of all column indices in the dataset. This list is used to randomly select columns for
        generating interactions.
        - interaction_terms (list or Series): List of column indices for which interaction terms are explicitly desired.
        This list guides the focus of interaction term generation.
        - important_variables (list or Series of int, optional): The subset of 'cols' that are actually used in calculation of target variable.
        If not provided, defaults to using 'cols'.
        - important_samples (int, optional): Number of samples to take from the important variables for each interaction term.
        If not provided, defaults to one-fifth of the length of 'targets'.
        - other_samples (int, optional): Number of samples to take from the set difference of 'cols' and 'targets' for each interaction term.
        If not provided, defaults to one-fifth of the difference in length between 'cols' and 'targets'.

        Returns:
        - Series: A Series where each index corresponds to an index of an interaction term, and the row is a list of functions.
        These functions, when applied, generate the interaction terms for the dataset, ready for use in further analysis or modeling.
        """
        # Set class values if parameters None
        cols = cols if cols is not None else self.cols
        interaction_terms = interaction_terms if interaction_terms is not None else self.interaction_terms
        important_variables = important_variables if important_variables is not None else self.important_variables
        correlation_range = correlation_range if correlation_range is not None else self.correlation_range

        interactions = {}

        # Get random columns to interact with
        for term in interaction_terms:
            mimicking = random.choice(important_variables)
            correlation = random.uniform(correlation_range[0], correlation_range[1])

            interactions[term] = (mimicking, correlation)

        return interactions

    def generate_data(self, num_rows=None, cols=None, frequencies=None, effects=None, interactions=None, target=None, intercept=None, noise=None):
        """
        Generates a complex dataset with binary columns, interaction terms, noise, and a target variable.
        This function allows for the simulation of datasets with specified properties, including
        predefined effects for certain columns, variable frequencies, interactions between variables,
        and a range of noise to simulate real-world data variance.

        Parameters:
        - num_rows (int): Number of rows (samples) in the generated dataset.
        - cols (list or Series): List of column indices that will be included in the dataset.
        - effects (dict, optional): Dictionary where keys are column indices and values are functions
        that define how each column influences the target variable.
        - frequencies (dict, optional): Dictionary specifying the frequency (probability) of 1s for each binary column.
        Keys are column indices, and values are probabilities (0 to 1).
        - interactions (dict or Series, optional): Dictionary specifying interactions between columns.
        Keys are column indices, and values are lists of functions representing the interaction effects.
        - target (str, optional): Name of the target column.
        - intercept (float, optional): The intercept (bias) term added to the target variable calculation.
        It can shift the target variable up or down.
        - noise (float, optional): Bound for the uniform distribution from which noise is generated (from -1 * noise to noise)

        Returns:
        - DataFrame: A pandas DataFrame containing the generated dataset. Includes binary columns as specified by 'cols',
        interaction terms as specified by 'interactions', and a target column influenced by 'effects', 'intercepts', and added noise.

        This function first generates binary data for each column based on specified frequencies.
        Then, it applies interaction functions to create complex relationships between variables.
        Noise is uniformly added to introduce variability.
        The target variable is calculated by summing the effects of important columns, interactions, and noise, adjusted by the intercept.
        This allows for the creation of datasets that can simulate various real-world scenarios, useful for testing machine learning models and data analysis techniques.
        """
        num_rows = num_rows if num_rows is not None else self.num_rows
        cols = cols if cols is not None else self.cols
        frequencies = frequencies if frequencies is not None else self.frequencies
        effects = effects if effects is not None else self.effects
        interactions = interactions if interactions is not None else self.interactions
        target = target if target is not None else self.target
        intercept = intercept if intercept is not None else self.intercept
        noise = noise if noise is not None else self.noise

        rng = self.rng
        data = {}

        # Generate data for each column
        for col in cols:
            freq = frequencies[col]
            data[col] = rng.choice([0, 1], size=num_rows, p=[1-freq, freq])

        df = pd.DataFrame(data)

        # Generate interactions
        for col in interactions.keys():
            mimicking, correlation = interactions[col]

            df[col] = df.apply(lambda row: (1 - row[mimicking]) if (random.uniform(0, 1) < abs(correlation) and correlation < 0) 
                    else (row[mimicking] if (random.uniform(0, 1) < abs(correlation) and correlation > 0) 
                        else row[col]), 
            axis=1)
            df[col] = df[col].astype(int)

        # Generate noise uniformly
        noise = rng.uniform(-(noise), noise, size=df.shape[0])

        # Calculate target variable
        important_sum = sum(df[col].apply(effects[col]) for col in df.columns if col != target)

        # Generate target based on important cols, interactions, and non-linear effects
        df[target] = important_sum + noise + intercept

        return df