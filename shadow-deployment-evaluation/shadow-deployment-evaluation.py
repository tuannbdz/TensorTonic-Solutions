import math
def accuracy(logs):
    acc = 0
    n = len(logs)
    for log in logs:
        if log['actual'] == log['prediction']:
            acc += 1
    return acc / n
def evaluate_shadow(production_log, shadow_log, criteria):
    """
    Evaluate whether a shadow model is ready for promotion.
    """
    # Write code here
    output = {}
    output['metrics'] = {}
    prod_acc = accuracy(production_log)
    sha_acc = accuracy(shadow_log)
    acc_gain = sha_acc - prod_acc
    n = len(production_log)
    agreement_rate = 0
    for i in range(n):
        if (production_log[i]['prediction'] == shadow_log[i]['prediction']):
            agreement_rate += 1
    agreement_rate /= n

    shadow_log = sorted(shadow_log, key=lambda x: x['latency_ms'])
    P95_latency = shadow_log[int(math.ceil(0.95 * n)) - 1]['latency_ms']
    if acc_gain >= criteria['min_accuracy_gain'] and P95_latency <= criteria['max_latency_p95'] and agreement_rate >= criteria['min_agreement_rate']:
        output['promote'] = True
    else:
        output['promote'] = False
    output['metrics']['shadow_accuracy'] = sha_acc
    output['metrics']['production_accuracy'] = prod_acc
    output['metrics']['accuracy_gain'] = acc_gain
    output['metrics']['shadow_latency_p95'] = P95_latency
    output['metrics']['agreement_rate'] = agreement_rate
    return output
        
    