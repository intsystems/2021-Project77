def DelayEmbeddingPhaseIdentifier(
    x,
    nlags = 500,
    epsilon = None,
    min_dim = 4,
    l1 = 1.0,
    l2 = 1.0,
    l3 = 1.0,
    n_upd = np.inf,
    make_plots = False,
    return_variance_model = False,
    return_expectation_model = False
):

    """
    TODO: DESCRIPTION
    """
    warnings.warn("The method is not optimized. Time complexity O(n^3)")

    # Autocorr and finding indeces of first period
    acf_result = acf(x[nlags-1:],
                     nlags = len(x),
                     fft = True)
    if make_plots:
        plt.figure(figsize=(15,5))
        plt.plot(acf_result)

    peaks_indices = find_peaks(acf_result)[0]
    peaks_height = acf_result[peaks_indices]
    i_indices = sorted(range(len(peaks_indices)), key=lambda i: peaks_height[i], reverse = True)[0]

    period_start, period_stop = 0, peaks_indices[i_indices]
    period = period_stop - period_start
    
    # PCA from initial phase space to lower dimention
    X_maxdim = HankelMatrix(x, nlags)
    pca = PCA(n_components = min_dim)
    X = pca.fit_transform(X_maxdim)

    # Making a model in phase space
    metric = lambda x,y: (1 - np.cos((x-y)))/2
    phase = np.linspace(0, 2 * np.pi, period).reshape((period, 1))
    delta_phase = 0.5 * float(phase[1] - phase[0])

    expectation = NWregression(h = delta_phase,
                               metric = metric)
    expectation.fit(phase,X[period_start : period_stop])
    expectation_array = expectation.predict(phase)

    variance_init = 0.25 * max(distance_matrix(expectation_array, expectation_array, p = 2).reshape(-1,))
    variance = NWregression(h = delta_phase,
                            metric = metric)
    variance.fit(phase,np.full((len(expectation_array),1), variance_init))
    variance_array = variance.predict(phase)

    # Area for history point
    if epsilon is None:
        epsilon = 0.5 * variance_init

    # Implementing of phase retrieval algo 
    history_phase = []
    history_x = []
    n_points = len(expectation_array)
    model_phases = np.linspace(0, 2 * np.pi, n_points)
    
    # Updating models parametres
    prev_update = 0
    
    for i in tqdm(np.arange(len(X))):
        # Nearest neigh at the beggining of alg
        if len(history_phase) == 0:
            model_indeces = np.argmin(distance_matrix(np.array([X[i]]), expectation_array, p = 2))
            current_phi = model_phases[model_indeces]
            history_x.append(X[0])
            history_phase.append(float(current_phi))
            continue

        # Nearest neigh at the approximation function
        model_indeces = np.array([j for j in range(n_points)
                                  if np.linalg.norm(X[i] - expectation_array[j]) <= variance_array[j]]) 


        # IF DIST IS TO BIG FOR CURRENT VARIANCE MODEL
        if len(model_indeces) == 0:
            model_indeces = np.argmin(distance_matrix(np.array([X[i]]), expectation_array, p = 2))
            possible_phi = np.array([model_phases[model_indeces]])
            VARIANCE = np.array([variance_array[model_indeces]])
            EXPECT = np.array([expectation_array[model_indeces]])
        else:
            possible_phi = model_phases[model_indeces]
            VARIANCE = variance_array[model_indeces]
            EXPECT = expectation_array[model_indeces]

        # Nearest neigh at the history
        near_from_history = np.array([history_phase[j] for j in range(len(history_x))
                                      if np.linalg.norm(X[i] - history_x[j]) <= epsilon])

        # Choosing phi acording to loss function
        idx_min = np.argmin(l1 * L1(history_phase[-1], possible_phi)\
                            + l2 * L2(near_from_history, possible_phi)\
                            + l3 * L3(X[i], EXPECT, VARIANCE)
                           )

        current_phi = possible_phi[idx_min]

        # Filling in history
        history_x.append(X[i])
        history_phase.append(float(current_phi))
        
        # Updates models
        if np.abs(history_phase[-1] -  2 * np.pi) < 0.02 and np.abs(i - prev_update) > (1/4 + n_upd)*period:
            
            prev_update = i.copy()
            
            expectation = NWregression(h = delta_phase,
                                       metric = metric)
            expectation.fit(np.array(history_phase).reshape((len(history_x), 1)),
                            np.array(history_x))
            expectation_array = expectation.predict(phase)

            variance_current = np.linalg.norm(np.array(history_x) - expectation\
                                              .predict(np.array(history_phase).reshape((len(history_x), 1))),
                                              axis = 1).reshape((len(history_x), 1))
            variance = NWregression(h = delta_phase,
                                    metric = metric)
            variance.fit(np.array(history_phase).reshape((-1, 1)),
                         variance_current)
            variance_array = 3 * variance.predict(phase)

    # Preparing results
    result_dict = {}
    result_dict['phase'] = np.array(history_phase)

    if     return_variance_model:
        result_dict['variance'] = variance_array

    if return_expectation_model:
        result_dict['expectation'] = expectation_array
    return result_dict
  
  
  def ro_cos(x):
    return (1 - np.cos(x))/2

def ro_chord(x,y):
    x = np.array([np.sin(x), np.cos(x)])
    y = np.array([np.sin(y), np.cos(y)])
    return np.linalg.norm(x-y)

def L1(previous_phi, phi):
    loss1 = np.zeros_like(phi)
    
    for i in range(len(phi)):
        loss1[i] = ro_cos(max(0,previous_phi - phi[i]))

    return loss1

def L2(near_phis, phi):
    loss2 = np.zeros_like(phi)
    
    for i in range(len(phi)):
        for near_phi in near_phis:
            loss2[i] += (1 - np.cos(near_phi - phi[i]))/2

    return loss2

def L3(x, x_neigh, normalization):
    x_array = np.full_like(x_neigh, x)
    loss3 = np.linalg.norm(x_array - x_neigh, axis = 1)/normalization.reshape(-1,)
    
    return loss3
