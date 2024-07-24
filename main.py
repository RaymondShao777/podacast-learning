import sys, os, json
sys.path.append(os.path.join(sys.path[0],'src'))
from plot import draw_heatmap, plot_embeddings
from util import encode_to_frame, cos_sim
from sentence_transformers import SentenceTransformer
import pandas as pd

def main():
    # defines model (we can change this to use other models)
    model = SentenceTransformer('all-mpnet-base-v2')

    # check that necessary files exist
    if not os.path.isfile(os.path.join(sys.path[0],'data/script.json')):
        sys.stdout.write('data/script.json is missing!')
        exit(1)
    if not os.path.isfile(os.path.join(sys.path[0],'data/data.json')):
        sys.stdout.write('data/data.json is missing!')
        exit(1)

    # read in data
    script = dict()
    responses = dict()
    with open(os.path.join(sys.path[0],'data/script.json')) as f:
        script = json.load(f)
    with open(os.path.join(sys.path[0],'data/data.json')) as f:
        responses = json.load(f)

    # generate pd DataFrames of sentence embeddings
    eva_av_responses = encode_to_frame(responses['evaporation']['audiovisual'], model)
    eva_a_responses = encode_to_frame(responses['evaporation']['audio'], model)
    eva_pictures = encode_to_frame(script['evaPic'], model)
    pre_av_responses = encode_to_frame(responses['precipitation']['audiovisual'], model)
    pre_a_responses = encode_to_frame(responses['precipitation']['audio'], model)
    pre_pictures = encode_to_frame(script['prePic'], model)

    # run cosine similarity (these will be DataFrames too)
    eva_av_cos_sim = cos_sim(eva_av_responses, eva_pictures)
    eva_a_cos_sim = cos_sim(eva_a_responses, eva_pictures)
    pre_av_cos_sim = cos_sim(pre_av_responses, pre_pictures)
    pre_a_cos_sim = cos_sim(pre_a_responses, pre_pictures)

#-------------------------------------------------------------------------------
# Visualization (TSNE)
#-------------------------------------------------------------------------------
    plot_embeddings(eva_av_responses, eva_a_responses, eva_pictures, 'Evaporation')
    plot_embeddings(pre_av_responses, pre_a_responses, pre_pictures, 'Precipitation')

#-------------------------------------------------------------------------------
# Cosine Similarity
#-------------------------------------------------------------------------------

    # find per-subject mean for cosine similarity between the two pictures
    eva_av_mean = eva_av_cos_sim.mean(axis=1)
    eva_a_mean = eva_a_cos_sim.mean(axis=1)
    pre_av_mean = pre_av_cos_sim.mean(axis=1)
    pre_a_mean = pre_a_cos_sim.mean(axis=1)

    # find the overall mean per condition
    mean_by_condition = pd.DataFrame({
        "Evaporation":[eva_a_mean.mean(), eva_av_mean.mean()],
        "Precipitation":[pre_a_mean.mean(), pre_av_mean.mean()]
    }, index=["Audio", "Audiovisual"])
    eva_mean = pd.concat([eva_a_mean, eva_av_mean])
    pre_mean = pd.concat([pre_a_mean, pre_av_mean])
    mean_by_subject = pd.concat([eva_mean, pre_mean], axis=1)
    mean_by_subject.columns = ["Evaporation", "Precipitation"]

    # draw heatmaps for both evaporation and precipitation images
    draw_heatmap(mean_by_condition, "Condition", "Water Cycle Processes", v_min=-0.020738222, v_max=0.68389034)
    draw_heatmap(mean_by_subject, "Sub ID", "Image Average")

if __name__ == '__main__':
    main()
