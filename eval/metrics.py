'''
Helper methods for quantitative analyses.
'''

from __init__ import *


def calculate_metrics_for(data_retval, model_retval):
    '''
    Calculates (preliminary) useful quantitative results for this (sub-)batch. All of these
        numbers are just for information and not for backpropagation.
    :param data_retval (dict).
    :param model_retval (dict).
    '''
    pass
    
    # Return results.
    metrics_retval = dict()

    for (k, v) in metrics_retval.items():
        v = torch.tensor(v, device=model_retval['output_mask'].device)
        if 'count' in k:
            v = v.type(torch.int32).detach()
        else:
            v = v.type(torch.float32).detach()
        metrics_retval[k] = v

    return metrics_retval


def calculate_weighted_averages(metrics_retvals):
    '''
    :param metrics_retvals (list) of metric_retval dicts, one per batch.
    :return (dict) of weighted averages.
    '''
    # TODX DRY: This is also in loss.py.
    final_metrics = dict()
    for k in metrics_retvals[0].keys():
        if 'count' in k:
            count_key = k
            mean_key = k.replace('count', 'mean')
            old_counts = np.array([x[count_key] for x in metrics_retvals])
            old_means = np.array([x[mean_key] for x in metrics_retvals])

            # NOTE: Some mean values will be -1.0 but then corresponding counts are always 0.
            new_count = old_counts.sum()
            if new_count > 0:
                new_mean = np.multiply(old_means, old_counts).sum() / (new_count + 1e-7)
            else:
                new_mean = -1.0
            final_metrics[count_key] = new_count
            final_metrics[mean_key] = new_mean

    return final_metrics


def calculate_unweighted_averages(metrics_retvals, exclude_value=-1.0):
    '''
    :param metrics_retvals (list) of metric_retval dicts, one per batch.
    :return (dict) of unweighted averages.
    '''
    final_metrics = dict()
    for k in metrics_retvals[0].keys():
        if 'count' in k:
            count_key = k
            mean_key = k.replace('count', 'mean')
            all_values = np.array([x[mean_key] for x in metrics_retvals])

            filtered_values = all_values[all_values != exclude_value]
            if len(filtered_values) > 0:
                mean_value = filtered_values.mean()
            else:
                mean_value = np.nan

            final_metrics[count_key] = len(filtered_values)
            final_metrics[mean_key] = mean_value

    return final_metrics


def test_results_to_dataframe(inference_retvals):
    test_results = defaultdict(list)

    for inference_retval in inference_retvals:
        cur_data_retval = inference_retval['data_retval_pruned']
        cur_loss_retval = inference_retval['loss_retval']
        cur_metrics_retval = inference_retval['loss_retval']['metrics']

        test_results['source'].append(cur_data_retval['source_name'][0])
        test_results['dset_idx'].append(cur_data_retval['dset_idx'].item())
        test_results['scene_idx'].append(cur_data_retval['scene_idx'].item())
        if 'scene_dn' in cur_data_retval:
            test_results['scene_dn'].append(cur_data_retval['scene_dn'][0])
        test_results['friendly_short_name'].append(inference_retval['friendly_short_name'])

        for (k, v) in cur_loss_retval.items():
            if not('metrics' in k):
                if torch.is_tensor(v):
                    v = v.item()
                test_results['loss_' + k].append(v)

        for (k, v) in cur_metrics_retval.items():
            test_results[k].append(v)

    test_results = pd.DataFrame(test_results)
    return test_results


def calculate_weighted_averages_dataframe(csv):
    '''
    :param csv (pd.df).
    :return (dict) of weighted averages.
    '''
    # TODX DRY: This is also in loss.py.
    final_metrics = dict()
    for k in csv.columns:
        if 'count' in k:
            count_key = k
            mean_key = k.replace('count', 'mean')
            old_counts = np.array(csv[count_key])
            old_means = np.array(csv[mean_key])

            # NOTE: Some mean values will be -1.0 but then corresponding counts are always 0.
            new_count = old_counts.sum()
            if new_count > 0:
                new_mean = np.multiply(old_means, old_counts).sum() / (new_count + 1e-7)
            else:
                new_mean = -1.0
            final_metrics[count_key] = new_count
            final_metrics[mean_key] = new_mean

    return final_metrics


def calculate_unweighted_averages_dataframe(csv, exclude_value=-1.0):
    '''
    :param csv (pd.df).
    :return (dict) of unweighted averages.
    '''
    final_metrics = dict()
    for k in csv.columns:
        if 'count' in k:
            count_key = k
            mean_key = k.replace('count', 'mean')
            all_values = np.array(csv[mean_key])

            filtered_values = all_values[all_values != exclude_value]
            if len(filtered_values) > 0:
                mean_value = filtered_values.mean()
            else:
                mean_value = np.nan

            final_metrics[count_key] = len(filtered_values)
            final_metrics[mean_key] = mean_value

    return final_metrics


def pretty_print_aggregated(logger, weighted_metrics, unweighted_metrics, num_scenes):

    longest_key = max([len(x) for x in weighted_metrics.keys()])

    logger.info()
    for k in sorted(weighted_metrics.keys()):
        if 'count' in k:
            count_key = k
            mean_key = k.replace('count', 'mean')
            short_key = k.replace('count_', '')
            final_count = weighted_metrics[count_key]
            unweighted_mean_value = unweighted_metrics[mean_key]
            logger.info(f'{("unweighted_" + mean_key).ljust(longest_key + 11)}  '
                        f'{(f"(over {num_scenes} scenes)").ljust(18)}:  '
                        f'{unweighted_mean_value:.5f}')
            if final_count > 0:
                logger.report_single_scalar('unweighted_' + short_key, unweighted_mean_value)

    logger.info()
    for k in sorted(weighted_metrics.keys()):
        if 'count' in k:
            count_key = k
            mean_key = k.replace('count', 'mean')
            short_key = k.replace('count_', '')
            final_count = weighted_metrics[count_key]
            weighted_mean_value = weighted_metrics[mean_key]
            logger.info(f'{("weighted_" + mean_key).ljust(longest_key + 8)}  '
                        f'{(f"(over {final_count} frames)").ljust(19)}:  '
                        f'{weighted_mean_value:.5f}')
            if final_count > 0:
                logger.report_single_scalar('weighted_' + short_key, weighted_mean_value)
