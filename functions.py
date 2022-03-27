def cat_onehot(df):
    """One hot encodes all categorical variables from a DataFrame. 
       - Cretes a new DataFrame with columns names = orginal categorical name + unqiue value (i.e engine_v6 or color_red)
       - Drops original categorical column 
    Args:
        df (pandas DataFrame): feature variable data frame
    """
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import LabelBinarizer
    enc = OneHotEncoder(handle_unknown = "ignore", sparse='False')
    label_enc = LabelBinarizer()

    for c in df.select_dtypes(exclude=['number']):
        # get matrix and names of new features
        mat = enc.fit_transform(df[[c]])
        names = enc.get_feature_names_out()
        label_enc.fit(df[c])
        # transform encoded 
        transformed = label_enc.transform(df[c]) 
        # create dataframe of new features with names as columns
        ohe_df = pd.DataFrame(transformed, columns=names)
        df = pd.concat([df, ohe_df], axis=1).drop([c], axis=1)
        
    return df

# Automating backward elimination technique

def DoBackwardElimination(the_regressor, X, y, X_val, minP2eliminate):
    """Performos backward elimination using linear regression.
       Output is a list of features kept after backward elimination and the
       corresponding input data set with updated list of features.

    Args:
        the_regressor (_type_): _description_
        X (DataFrame): _description_
        y (Series): _description_
        X_val (DataFrame): _description_
        minP2eliminate (float): _description_

    Returns:
        _type_: _description_
    """
    
    assert np.shape(X)[0] == np.shape(y)[0], 'Length of X and y do not match'
    assert minP2eliminate > 0, 'Minimum P value to eliminate cannot be zero or negative'
    
    original_list = list(range(0, np.shape(the_regressor.pvalues)[0]))
    
    max_p = 10        # Initializing with random value of maximum P value
    i = 0
    r2adjusted = []   # Will store R Square adjusted value for each loop
    r2 = []           # Will store R Square value  for each loop
    list_of_originallist = [] # Will store modified index of X at each loop
    classifiers_list = [] # fitted classifiers at each loop
    
    while max_p >= minP2eliminate:
        
        p_values = list(the_regressor.pvalues)
        r2adjusted.append(the_regressor.rsquared_adj)
        r2.append(the_regressor.rsquared)
        list_of_originallist.append(original_list)
        
        max_p = max(p_values)
        max_p_idx = p_values.index(max_p)
                
        if max_p < minP2eliminate:
            
            print('Max P value found less than ', str(minP2eliminate), ' without 0 index...Loop Ends!!')
            
            break
        
        val_at_idx = original_list[max_p_idx]
        
        idx_in_org_lst = original_list.index(val_at_idx)
        
        original_list.remove(val_at_idx)
        
        print('Popped column index out of original array is {} with P-Value {}'.format(val_at_idx, np.round(np.array(p_values)[max_p_idx], decimals= 4)))
        
        X_new = X.iloc[:,original_list]
        X_val_new = X_val.iloc[:,original_list]
        
        the_regressor = smf.OLS(endog = y, exog = X_new).fit()
        classifiers_list.append(the_regressor)
        
        print('==================================================================================================')
        
    return classifiers_list, r2, r2adjusted, list_of_originallist, X_new, X_val_new

def Calculate_Error(original_values, predicted_values):
    assert len(original_values) == len(predicted_values), 'Both list should have same length'
    temp = 0
    error = 0
    n = len(original_values)
    for o, p in zip(original_values, predicted_values):
        temp = temp + ((o-p)**2)
        
    temp = temp/n
    error = np.sqrt(temp)
    return error