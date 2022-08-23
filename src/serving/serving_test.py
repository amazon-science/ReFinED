from serving.serve import disamb

# Tests an example request from Gerbil entity linking benchmarking framework.
# https://github.com/dice-group/gerbil
# To run this an intermediate server must be setup to forward requests from Gerbil runner to this Flask service.
# Code and instructions for this can be found here:
# https://github.com/dalab/end2end_neural_el/tree/master/gerbil-SpotWrapNifWS4Test.


request_dict_eg_1 = {
    "text": "NCAA AMERICAN FOOTBALL-OHIO STATE'S PACE FIRST REPEAT LOMBARDI AWARD WINNER. HOUSTON "
    "1996-12-05 Ohio State left tackle Orlando Pace became the first repeat winner of the "
    "Lombardi Award Thursday night when the Rotary Club of Houston again honoured him as "
    "college football's lineman of the year. Pace, a junior, helped Ohio State to a 10-1 "
    "record and a berth in the Rose Bowl against Arizona State. He was the most dominant "
    "offensive lineman in the country and also played defensive line in goal-line "
    "situations. Last year, Pace became the first sophomore to win the award since its "
    "inception in 1970. Pace outdistanced three senior finalists-- Virginia Tech defensive "
    "end Cornell Brown, Arizona State offensive tackle Juan Roque and defensive end Jared "
    "Tomich of Nebraska. The Lombardi Award is presented to the college lineman who, "
    "in addition to outstanding effort on the field, best exemplifies the characteristics "
    "and discipline of Vince Lombardi, legendary coach of the Green Bay Packers.",
    "spans": [
        {"startPosition": 600, "length": 4},
        {"startPosition": 54, "length": 14},
        {"startPosition": 14, "length": 19},
        {"startPosition": 382, "length": 13},
        {"startPosition": 77, "length": 7},
        {"startPosition": 209, "length": 11},
        {"startPosition": 671, "length": 13},
        {"startPosition": 294, "length": 4},
        {"startPosition": 170, "length": 14},
        {"startPosition": 0, "length": 4},
        {"startPosition": 317, "length": 10},
        {"startPosition": 717, "length": 10},
        {"startPosition": 746, "length": 12},
        {"startPosition": 776, "length": 14},
        {"startPosition": 762, "length": 8},
        {"startPosition": 974, "length": 17},
        {"startPosition": 96, "length": 10},
        {"startPosition": 364, "length": 9},
        {"startPosition": 224, "length": 7},
        {"startPosition": 119, "length": 12},
        {"startPosition": 36, "length": 4},
        {"startPosition": 522, "length": 4},
        {"startPosition": 643, "length": 13},
        {"startPosition": 686, "length": 13},
        {"startPosition": 935, "length": 14},
    ],
}

request_dict_eg_2 = {
    "text": "CYCLING - BALLANGER KEEPS SPRINT TITLE IN STYLE. Martin Ayres MANCHESTER, England 1996-08-30 Felicia "
    "Ballanger of France confirmed her status as the world's number one woman sprinter when she retained her "
    "title at the world cycling championships on Friday. Ballanger beat Germany's Annett Neumann 2-0 in the "
    "best-of-three matches final to add the world title to the Olympic gold medal she won in July. France "
    "also took third place in the sprint, Magali Faure defeating ex-world champion Tanya Dubnicoff of Canada "
    "2-0. Ballanger, 25, will be aiming to complete a track double when she defends her 500 metres time trial "
    "title on Saturday. The other final of the night, the women's 24-kms points race, also ended in success "
    "for the reigning champion. Russia's Svetlana Samokhalova fought off a spirited challenge from American "
    "Jane Quigley to take the title for a second year. Russia, the only nation to have two riders in the "
    "field, made full use of their numerical superiority. Goulnara Fatkoullina helped Samokhalova to build an "
    "unbeatable points lead before snatching the bronze medal. Quigley, a former medallist in the points "
    'event, led the race at half distance. "I went so close this time, but having two riders certainly gave '
    'the Russians an advantage," she said. The first six riders lapped the field, which left former world '
    "champion Ingrid Haringa of the Netherlands down in seventh place despite having the second highest "
    "points score. Olympic champion Nathalie Lancien of France also missed the winning attack and finished a "
    "disappointing 10th.",
    "spans": [
        {"startPosition": 93, "length": 17},
        {"startPosition": 366, "length": 7},
        {"startPosition": 518, "length": 9},
        {"startPosition": 446, "length": 12},
        {"startPosition": 757, "length": 20},
        {"startPosition": 49, "length": 12},
        {"startPosition": 874, "length": 6},
        {"startPosition": 1236, "length": 8},
        {"startPosition": 506, "length": 6},
        {"startPosition": 257, "length": 9},
        {"startPosition": 62, "length": 10},
        {"startPosition": 282, "length": 14},
        {"startPosition": 487, "length": 15},
        {"startPosition": 114, "length": 6},
        {"startPosition": 977, "length": 20},
        {"startPosition": 272, "length": 7},
        {"startPosition": 824, "length": 12},
        {"startPosition": 748, "length": 6},
        {"startPosition": 402, "length": 6},
        {"startPosition": 1005, "length": 11},
        {"startPosition": 1342, "length": 14},
        {"startPosition": 815, "length": 8},
        {"startPosition": 1463, "length": 16},
        {"startPosition": 1087, "length": 7},
        {"startPosition": 10, "length": 9},
        {"startPosition": 1483, "length": 6},
        {"startPosition": 1364, "length": 11},
        {"startPosition": 1446, "length": 7},
        {"startPosition": 74, "length": 7},
    ],
}


def test_gerbil_request_does_not_throw_exception():
    disamb(request_dict_eg_1)
    disamb(request_dict_eg_2)
