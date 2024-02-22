import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        # Check that input is not length zero
        if input_observation_states.size == 0:
            raise ValueError("Input must have length of at least 1")
        
        #Check that all inputs are possible emissions
        if set(self.observation_states).issuperset(set(input_observation_states))== False:
            raise ValueError("Not all emissions are present in observation states")

        # Step 1. Initialize variables
        result_matrix =np.ndarray((len(self.hidden_states),len(input_observation_states)))
        
        #Set inital values
        result_matrix[:,0]=self.prior_p * self.emission_p[:,self.observation_states_dict[input_observation_states[0]]]

        
        # Step 2. Calculate probabilities
        for i in range(1,len(input_observation_states)):
            #Calculated teh emission proablities given teh previous states
            probmatrix=self.transition_p * self.emission_p[:,self.observation_states_dict[input_observation_states[i]]]
            
            # Multiply by the previous probalities
            joint_prob=result_matrix[:,i-1]* probmatrix.T
    
            #sum columns to get total probablity of hidden state and assing to results matrix
            result_matrix[:,i]=np.sum(joint_prob, axis=1).T
         

        

        # Step 3. Return final probability
        return sum(result_matrix[:,len(input_observation_states)-1])


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        # Check that input is not length zero
        if decode_observation_states.size == 0:
            raise ValueError("Input must have length of at least 1")
        
        #Check that all inputs are possible emissions
        if set(self.observation_states).issuperset(set(decode_observation_states))== False:
            raise ValueError("Not all emissions are present in observation states")
        # Step 1. Initialize variables
        
        #store probabilities of hidden state at each step 
        viterbi_table = np.zeros((len(decode_observation_states), len(self.hidden_states)))
        #store best path for traceback
        backpointer_table = np.zeros((len(decode_observation_states), len(self.hidden_states)), dtype=int)
                
       # Initalize based on priors
        
        viterbi_table[0] = self.prior_p * self.emission_p[:,self.observation_states_dict[decode_observation_states[0]]]
        for i in range( 0,len(self.hidden_states)):
            backpointer_table[0,i]=i

        # Step 2. Calculate Probabilities
        for i in range(1, len(decode_observation_states)):
            for j in range(len(self.hidden_states)):
                # Calculate the probabilities for each state at time i
                probabilities = viterbi_table[i - 1] * self.transition_p[:, j] * self.emission_p[j,self.observation_states_dict[decode_observation_states[i]]]

                # Select the maximum probability and store it in the viterbi table
                viterbi_table[i, j] = np.max(probabilities)
                
                # Store the backpointer which indicates the previous state that led to this maximum probability
                backpointer_table[i, j] = np.argmax(probabilities)
            
        # Step 3. Traceback 
        # id best state at end         
        best_hidden_state_sequence = [np.argmax(viterbi_table[-1])]
        # Create reverse path based on backpointer table
        for t in range(len(decode_observation_states) - 1, 0, -1):
            best_hidden_state_sequence.insert(0, backpointer_table[t, best_hidden_state_sequence[0]])
        # Step 4. Return best hidden state sequence (revseve since it was backtraced)
        return [self.hidden_states_dict[x]for x in best_hidden_state_sequence]

   
   
    
