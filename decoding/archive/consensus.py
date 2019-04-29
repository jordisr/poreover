def pair_forward(l, y1, y2, mask=None, previous=None):
    '''
    Naive implementation.
    Alphas are stored in dense numpy arrays, iteration is done over all SxUxV
    elements, though if an alignment envelope is present, only values are
    calculated for entries in the envelope (better soultion would be to just
    iterate directly over entries in envelope).
    '''
    U = len(y1)
    V = len(y2)
    S = len(l)
    shift = 2

    alpha = np.zeros(shape=(S+shift,U+shift,V+shift))
    alpha_ast_ast = np.zeros(shape=(S+shift,U+shift,V+shift))
    alpha_ast = np.zeros(shape=(S+shift,U+shift,V+shift))

    if previous is not None:
        alpha[:-1] = previous[0]
        alpha_ast[:-1] = previous[1]
        alpha_ast_ast[:-1] = previous[2]
        s_range = [S + shift - 1]
    else:
        s_range = range(1,S+shift)

    for s in s_range:
        for u in range(1,U+shift):
            for v in range(1,V+shift):
                alpha_eps = 0
                alpha_ast_eps = 0
                if s==1 and u==1 and v==1:
                    alpha_ast_ast[s,u,v] = 1
                elif (s==1 and (u==1 or v==1)) or (mask is None) or ((mask is not None) and ((u-shift,v-shift) in mask)):
                    alpha_eps = y1[u-shift,-1]*alpha[s,u-1,v]
                    alpha_ast_eps = y2[v-shift,-1]*alpha_ast[s,u,v-1]
                    alpha_ast_ast[s,u,v] = y1[u-shift,l[s-shift]]*y2[v-shift,l[s-shift]]*alpha[s-1,u-1,v-1]

                alpha_ast[s,u,v] = alpha_ast_ast[s,u,v] + alpha_ast_eps
                alpha[s,u,v] = alpha_eps + alpha_ast[s,u,v]

    return(alpha, alpha_ast_ast, alpha_ast)

def pair_label_prob(alpha):
    '''
    Returns last element, corresponding to label probability
    '''
    return(alpha[-1,-1,-1])

def pair_prefix_prob(alpha_ast_ast,gamma, envelope=None):
    U,V = gamma.shape
    prefix_prob = 0
    if envelope == None:
        for u in range(2,U+1):
            for v in range(2,V+1):
                prefix_prob += alpha_ast_ast[-1,u,v]*gamma[u-1,v-1]
    else:
        for k in envelope.keys():
            (u,v) = k
            prefix_prob += alpha_ast_ast[-1,u+2,v+2]*gamma[u+1,v+1]
    return(prefix_prob)

def pair_prefix_search(y1, y2, envelope=None, alphabet=DNA_alphabet, forward_algorithm=pair_forward):
    '''
    Do 2d prefix search. Arguments are softmax probabilities of each read,
    an alignment_envelope object, and an OrderedDict with the alphabet.
    Additionally, takes forward algorithm function (for testing different
    implementations).
    '''
    gamma_ = pair_gamma(y1,y2)

    stop_search = False
    search_level = 0

    # blank label prob is joint probability of both reads producing blanks
    gap_prob = np.product(y1[:,-1])*np.product(y2[:,-1])
    label_prob = {'':gap_prob}

    top_label = ''
    curr_label = ''
    curr_label_alphas = []

    while not stop_search:
        prefix_prob = {}
        prefix_alphas = []

        for c in alphabet:
            prefix = curr_label + c
            prefix_int = [alphabet[i] for i in prefix]

            if search_level == 0:
                prefix_alphas.append(forward_algorithm(prefix_int,y1,y2,mask=envelope))
            else:
                prefix_alphas.append(forward_algorithm(prefix_int, y1,y2, mask=envelope, previous=curr_label_alphas))

            label_prob[prefix] = pair_label_prob(prefix_alphas[-1][0])/gamma_[0,0]
            prefix_prob[prefix] = pair_prefix_prob(prefix_alphas[-1][1], gamma_, envelope=envelope)/gamma_[0,0]

            print(search_level, 'extending by prefix:',c, 'Prefix Probability:',prefix_prob[prefix], 'Label probability:',label_prob[prefix], file=sys.stderr)

        best_prefix = max(prefix_prob.items(), key=operator.itemgetter(1))[0]

        if prefix_prob[best_prefix] < label_prob[top_label]:
            stop_search = True
        else:
            # get highest probability label
            top_label = max(label_prob.items(), key=operator.itemgetter(1))[0]
            # then move to prefix with highest label probability
            curr_label = best_prefix
            curr_label_alphas = prefix_alphas[alphabet[curr_label[-1]]]

        search_level += 1

    return(top_label, label_prob[top_label])
