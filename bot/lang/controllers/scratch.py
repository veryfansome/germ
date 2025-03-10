

async def scratch():
    ## Noun grouping
    #idx_to_noun_group = {}
    #idx_to_noun_joined_base_form = {}
    #for noun_group in extract_label_idx_groups(ud_token_labels, "xpos", target_labels=NOUN_LABELS):
    #    grp_stop_idx = noun_group[-1]

    #    # grp_base_form_tokens = []
    #    # grp_deprel_labels = []
    #    # grp_noun_labels = []
    #    # grp_ner_labels = []
    #    # grp_raw_tokens = []
    #    # for idx in noun_group:
    #    #    deprel_label = ud_token_labels["deprel"][idx]
    #    #    grp_deprel_labels.append(deprel_label)

    #    #    noun_label = ud_token_labels["xpos"][idx]
    #    #    grp_noun_labels.append(noun_label)

    #    #    token = ud_token_labels["tokens"][idx]
    #    #    # If 1st token, not proper noun, and has only one upper character, convert to lower
    #    #    if idx == 0 and "P" not in noun_label and sum(ch.isupper() for ch in token) == 1:
    #    #        token = token.lower()

    #    #    grp_base_form_tokens.append(get_noun_base_form(token, noun_label).lower())
    #    #    grp_raw_tokens.append(token)

    #    #    if is_conll_ud_aligned:
    #    #        grp_ner_labels.append(conll_token_labels["ner_tags"][idx])

    #    ## The nominal subject that acts on the object shouldn't be in the same phrase entity
    #    # if ("nsubj" in ud_token_labels["deprel"][grp_start_idx:grp_stop_idx]
    #    #        and "obj" in ud_token_labels["deprel"][grp_start_idx:grp_stop_idx]):
    #    #    continue
    #    #    if joined_base_form not in nouns_added:
    #    #        await self.control_plane.add_noun(joined_base_form, sentence_id)
    #    #        await self.control_plane.add_noun_form(joined_base_form, joined_raw_form, sentence_id)
    #    #        nouns_added.add(joined_base_form)

    #    #    # If there's a proper noun, everything becomes proper
    #    #    noun_label = "NNP" if "NNP" in grp_noun_labels else "NN"
    #    #    # If the last noun is plural, everything becomes plural
    #    #    if grp_noun_labels[-1].endswith("S"):
    #    #        noun_label += "S"

    #    #    #await self.control_plane.link_noun_form_to_sentence(
    #    #    #    joined_base_form, joined_raw_form,
    #    #    #    noun_label,
    #    #    #    sentence_id)
    #    #    if "conj" in grp_deprel_labels:
    #    #        await self.control_plane.link_noun_form_to_sentence(
    #    #            joined_base_form, joined_raw_form,
    #    #            "conj",
    #    #            sentence_id)
    #    #    elif "nsubj" in grp_deprel_labels:
    #    #        await self.control_plane.link_noun_form_to_sentence(
    #    #            joined_base_form, joined_raw_form,
    #    #            "nsubj",
    #    #            sentence_id)
    #    #    elif "obj" in grp_deprel_labels:
    #    #        await self.control_plane.link_noun_form_to_sentence(
    #    #            joined_base_form, joined_raw_form,
    #    #            "obj",
    #    #            sentence_id)

    #    ## If CoNLL and UD tokens are aligned, graph NER entities and phrase components, else just graph
    #    ## phrase components.
    #    # if is_conll_ud_aligned:
    #    #    for entity_group in extract_entity_idx_groups(grp_ner_labels):
    #    #        entity_label, member_idx_list = entity_group

    #    #        member_base_form_tokens = [grp_base_form_tokens[i] for i in member_idx_list]
    #    #        member_raw_form_tokens = [grp_raw_tokens[i] for i in member_idx_list]
    #    #        joined_member_base_form = " ".join(member_base_form_tokens)
    #    #        joined_member_raw_form = " ".join(member_raw_form_tokens)

    #    #        # Add if phrase component is new
    #    #        if joined_member_base_form not in nouns_added:
    #    #            await self.control_plane.add_noun(joined_member_base_form, sentence_id)
    #    #            await self.control_plane.add_noun_form(
    #    #                joined_member_base_form, joined_member_raw_form, sentence_id)
    #    #            nouns_added.add(joined_member_base_form)
    #    #        if entity_label != "O":
    #    #            logger.info(f"ner entity group: {entity_group}: {joined_member_base_form}")
    #    #            entity_label_lowered = entity_label.lower()
    #    #            if entity_label_lowered not in noun_classes_added:
    #    #                await self.control_plane.add_noun_class(entity_label_lowered)
    #    #                noun_classes_added.add(entity_label_lowered)
    #    #            await self.control_plane.link_noun_to_noun_class(
    #    #                joined_member_base_form, entity_label_lowered, sentence_id)
    #    # else:
    #    #    for idx in range(len(grp_base_form_tokens)):
    #    #        noun_base_form = grp_base_form_tokens[idx]
    #    #        raw_token = grp_raw_tokens[idx]
    #    #        if noun_base_form not in nouns_added:
    #    #            await self.control_plane.add_noun(noun_base_form, sentence_id)
    #    #            await self.control_plane.add_noun_form(noun_base_form, raw_token, sentence_id)
    #    #            nouns_added.add(noun_base_form)
    #    #        await self.control_plane.link_noun_to_phrase(noun_base_form, joined_base_form, sentence_id)

    #    joined_base_form = None
    #    for idx in noun_group:
    #        # Look for joined base form from idx to end of group, if not already found
    #        if (joined_base_form is None
    #                and is_plausible_noun_phrase(ud_token_labels["deprel"][idx:grp_stop_idx])):
    #            joined_base_form = " ".join([get_noun_base_form(
    #                ud_token_labels["tokens"][idx], ud_token_labels["xpos"][idx]
    #            ).lower() for idx in noun_group])
    #        if joined_base_form is not None:
    #            # Map each idx to group and joined form
    #            idx_to_noun_group[idx] = noun_group[idx:grp_stop_idx]
    #            idx_to_noun_joined_base_form[idx] = joined_base_form
    #        else:
    #            # Map this idx to this token's base form
    #            idx_to_noun_group[idx] = [idx]
    #            idx_to_noun_joined_base_form[idx] = get_noun_base_form(
    #                ud_token_labels["tokens"][idx], ud_token_labels["xpos"][idx]).lower()

    ## Noun graphing
    #for pattern_name, start_positions in extract_consecutive_token_patterns(
    #        # Simplify noun labels for pattern matching
    #        simplify_pos_labels(ud_token_labels["xpos"]),
    #        [
    #            # Determiner, noun
    #            ["DT", "NN"],
    #            # Determiner, adjective, noun
    #            ["DT", "JJ", "NN"],
    #        ]
    #).items():
    #    if pattern_name in {"DT-NN"}:
    #        for start_idx in start_positions:
    #            det_lowered = ud_token_labels["tokens"][start_idx].lower()
    #            noun_base_form = idx_to_noun_joined_base_form[start_idx + 1]
    #            await self.add_det_noun(det_lowered, noun_base_form, sentence_id)
    #    elif pattern_name in {"DT-JJ-NN"}:
    #        for start_idx in start_positions:
    #            det_lowered = idx_to_noun_joined_base_form[start_idx].lower()
    #            noun_base_form = idx_to_noun_joined_base_form[start_idx + 2]
    #            await self.add_det_noun(det_lowered, noun_base_form, sentence_id)

    ## Pattern matching
    #for pattern_name, start_positions in extract_consecutive_token_patterns(
    #        # Simplify noun labels for pattern matching
    #        simplify_pos_labels(ud_token_labels["xpos"]),
    #        [
    #            # Attributive adjective, noun
    #            ["JJ", "NN"],
    #            # Noun possesses noun
    #            ["NN", "POS", "JJ", "NN"],
    #            ["NN", "POS", "NN"],
    #            # Noun, is/verb, a/the/the/that/.., noun
    #            ["NN", "VB", "DT", "NN"],
    #            # Noun, is/verb, noun
    #            ["NN", "VB", "NN"],
    #            # Noun, is/am/are, adjective
    #            ["NN", "VB", "JJ"],
    #        ]
    #).items():
    #    if pattern_name in {"JJ-NN"}:
    #        for start_idx in start_positions:
    #            jj = ud_token_labels["tokens"][start_idx]
    #            jj_lowered = jj.lower()
    #            if jj_lowered not in self.adjectives_added:
    #                await self.control_plane.add_adjective(jj_lowered)
    #                await self.control_plane.add_adjective_form(jj_lowered, jj)
    #                self.adjectives_added.add(jj_lowered)
    #            await self.control_plane.link_noun_to_adjective(
    #                jj_lowered, idx_to_noun_joined_base_form[start_idx + 1], sentence_id)
    #    elif pattern_name in {"NN-POS-JJ-NN"}:
    #        for start_idx in start_positions:
    #            await self.control_plane.link_noun_to_possessor(
    #                idx_to_noun_joined_base_form[start_idx], idx_to_noun_joined_base_form[start_idx + 3],
    #                sentence_id)
    #    elif pattern_name in {"NN-POS-NN"}:
    #        for start_idx in start_positions:
    #            await self.control_plane.link_noun_to_possessor(
    #                idx_to_noun_joined_base_form[start_idx], idx_to_noun_joined_base_form[start_idx + 2],
    #                sentence_id)
    #    elif pattern_name in {"NN-VB-NN"}:
    #        for start_idx in start_positions:
    #            if ud_token_labels["tokens"][start_idx + 1] in {"am", "are", "is"}:
    #                if ("P" in ud_token_labels["xpos"][start_idx]
    #                        and "P" in ud_token_labels["xpos"][start_idx + 2]):
    #                    # 1st noun is proper and 2nd noun is proper
    #                    await self.control_plane.equate_noun_to_noun(
    #                        idx_to_noun_joined_base_form[start_idx], idx_to_noun_joined_base_form[start_idx + 2],
    #                        sentence_id)
    #                elif "P" in ud_token_labels["xpos"][start_idx]:
    #                    pattern_cls = idx_to_noun_joined_base_form[start_idx + 2]
    #                    if pattern_cls not in self.noun_classes_added:
    #                        await self.control_plane.add_noun_class(pattern_cls)
    #                        self.noun_classes_added.add(pattern_cls)
    #                    await self.control_plane.link_noun_to_noun_class(
    #                        idx_to_noun_joined_base_form[start_idx], pattern_cls,
    #                        sentence_id)
    #                else:
    #                    pattern_cls1 = idx_to_noun_joined_base_form[start_idx]
    #                    if pattern_cls1 not in self.noun_classes_added:
    #                        await self.control_plane.add_noun_class(pattern_cls1)
    #                        self.noun_classes_added.add(pattern_cls1)
    #                    pattern_cls2 = idx_to_noun_joined_base_form[start_idx + 2]
    #                    if pattern_cls2 not in self.noun_classes_added:
    #                        await self.control_plane.add_noun_class(pattern_cls2)
    #                        self.noun_classes_added.add(pattern_cls2)
    #                    await self.control_plane.link_noun_cls1_is_cls2(
    #                        pattern_cls1, pattern_cls2)
    #            else:
    #                # TODO: Verbs also take many forms
    #                # create == made == crafted, etc.
    #                await self.control_plane.link_nouns_via_verb(
    #                    idx_to_noun_joined_base_form[start_idx],
    #                    wordnet_lemmatizer.lemmatize(ud_token_labels["tokens"][start_idx + 1], pos="v"),
    #                    idx_to_noun_joined_base_form[start_idx + 2],
    #                    sentence_id)
    #    elif pattern_name in {"NN-VB-DT-NN"}:
    #        for start_idx in start_positions:
    #            if (ud_token_labels["tokens"][start_idx + 1] in {"am", "are", "is"}
    #                    and ud_token_labels["tokens"][start_idx + 2] in {"a", "an"}):
    #                pattern_noun_class = idx_to_noun_joined_base_form[start_idx + 3]
    #                if pattern_noun_class not in self.noun_classes_added:
    #                    await self.control_plane.add_noun_class(pattern_noun_class)
    #                    self.noun_classes_added.add(pattern_noun_class)
    #                await self.control_plane.link_noun_to_noun_class(
    #                    idx_to_noun_joined_base_form[start_idx], pattern_noun_class,
    #                    sentence_id)
    #            else:
    #                await self.control_plane.link_nouns_via_verb(
    #                    idx_to_noun_joined_base_form[start_idx],
    #                    wordnet_lemmatizer.lemmatize(ud_token_labels["tokens"][start_idx + 1], pos="v"),
    #                    idx_to_noun_joined_base_form[start_idx + 3],
    #                    sentence_id)
    #    elif pattern_name in {"NN-VB-DT-JJ-NN"}:
    #        for start_idx in start_positions:
    #            if (ud_token_labels["tokens"][start_idx + 1] in {"am", "are", "is"}
    #                    and ud_token_labels["tokens"][start_idx + 2] in {"a", "an"}):
    #                pattern_noun_class = idx_to_noun_joined_base_form[start_idx + 4]
    #                if pattern_noun_class not in self.noun_classes_added:
    #                    await self.control_plane.add_noun_class(pattern_noun_class)
    #                    self.noun_classes_added.add(pattern_noun_class)
    #                await self.control_plane.link_noun_to_noun_class(
    #                    idx_to_noun_joined_base_form[start_idx], pattern_noun_class,
    #                    sentence_id)
    #            else:
    #                await self.control_plane.link_nouns_via_verb(
    #                    idx_to_noun_joined_base_form[start_idx],
    #                    wordnet_lemmatizer.lemmatize(ud_token_labels["tokens"][start_idx + 1], pos="v"),
    #                    idx_to_noun_joined_base_form[start_idx + 4],
    #                    sentence_id)
    #    elif pattern_name in {"NN-VB-JJ"}:
    #        for start_idx in start_positions:
    #            if ud_token_labels["tokens"][start_idx + 1] in {"is", "am", "are"}:
    #                jj = ud_token_labels["tokens"][start_idx + 2]
    #                jj_lowered = jj.lower()
    #                if jj_lowered not in self.adjectives_added:
    #                    await self.control_plane.add_adjective(jj_lowered)
    #                    await self.control_plane.add_adjective_form(jj_lowered, jj)
    #                    self.adjectives_added.add(jj_lowered)
    #                await self.control_plane.link_noun_to_adjective(
    #                    jj_lowered, idx_to_noun_joined_base_form[start_idx], sentence_id)

    ###
    ## Sequence walking

    #chunk_contexts = []
    #for chunk_positions in extract_sentence_chunks(ud_token_labels["xpos"]):
    #    chunk_context = {
    #        "nsubj": "",
    #        "obj": "",
    #    }
    #    for idx in chunk_positions:
    #        token = ud_token_labels["tokens"][idx]
    #        token_lowered = token.lower()
    #        ud_deprel_label = ud_token_labels["deprel"][idx]
    #        ud_xpos_label = ud_token_labels["xpos"][idx]
    #    chunk_contexts.append(chunk_context)

    ## graphed_nsubj_root_obj_link = False
    ## last_adj_idx = None
    ## last_adj_base = ""
    ## last_adj_token = ""
    ## last_noun_idx = None
    ## last_nsubj_noun_idx = None
    ## last_nsubj_noun = ""
    ## last_obj_idx = None
    ## last_obj_noun = ""
    ## last_root_verb = ""
    ## last_root_verb_idx = None
    ## last_verb = ""
    ## last_verb_idx = None
    ## last_walked_idx = None
    ## for idx in chunk_positions:
    ##    token = ud_token_labels["tokens"][idx]
    ##    lowered_token = token.lower()
    ##    deprel_label = ud_token_labels["deprel"][idx]
    ##    ud_xpos_label = ud_token_labels["xpos"][idx]

    ##    if ud_xpos_label in NOUN_LABELS:
    ##        # TODO:
    ##        # - Proper nouns should link to common nouns with INSTANCE_OF links
    ##        if (last_walked_idx is not None
    ##                and last_adj_base
    ##                and last_adj_idx == last_walked_idx):
    ##            # Link attributed adjectives that come immediately before nouns.
    ##            if last_adj_base not in adjectives_added:
    ##                await self.control_plane.add_adjective(last_adj_base)
    ##                await self.control_plane.add_adjective_form(last_adj_base, last_adj_token)
    ##                adjectives_added.add(last_adj_base)
    ##            await self.control_plane.link_noun_to_preceding_adjective(
    ##                last_adj_base, idx_to_noun_joined_base_form[idx])
    ##        if deprel_label == "nsubj":
    ##            last_nsubj_noun = lowered_token
    ##            last_nsubj_noun_idx = idx
    ##        elif deprel_label == "obj":
    ##            last_obj_noun = lowered_token
    ##            last_obj_idx = idx
    ##        last_noun_idx = idx

    ##    elif ud_xpos_label == "JJ":
    ##        last_adj_idx = idx
    ##        last_adj_base = lowered_token
    ##        last_adj_token = token

    ##    elif ud_xpos_label == "IN":

    ##        # Prepositions linking nouns
    ##        if lowered_token == "concerning":
    ##            lowered_token = "about"
    ##        elif lowered_token == "regarding":
    ##            lowered_token = "about"
    ##        if lowered_token in {"about", "from", "of", "in"}:
    ##            if 0 < idx:
    ##                if idx < token_cnt-1:
    ##                    if (ud_token_labels["xpos"][idx-1] in NOUN_LABELS
    ##                            and ud_token_labels["xpos"][idx+1] in NOUN_LABELS):
    ##                        # NN of NN
    ##                        await self.control_plane.link_nouns_via_preposition(
    ##                            idx_to_noun_joined_base_form[idx-1],
    ##                            token,
    ##                            idx_to_noun_joined_base_form[idx+1])
    ##                elif idx < token_cnt-2:
    ##                    if (ud_token_labels["xpos"][idx-1] in NOUN_LABELS
    ##                            and ud_token_labels["xpos"][idx+1] in {"DT", "JJ", "PRP$"}
    ##                            and ud_token_labels["xpos"][idx+2] in NOUN_LABELS):
    ##                        # NN of DT/JJ/PRP$ NN
    ##                        await self.control_plane.link_nouns_via_preposition(
    ##                            idx_to_noun_joined_base_form[idx-1],
    ##                            token,
    ##                            idx_to_noun_joined_base_form[idx+2])
    ##                elif idx < token_cnt-3:
    ##                    if (ud_token_labels["xpos"][idx-1] in NOUN_LABELS
    ##                            and ud_token_labels["xpos"][idx+1] in {"DT", "PRP$"}
    ##                            and ud_token_labels["xpos"][idx+2] == "JJ"
    ##                            and ud_token_labels["xpos"][idx+3] in NOUN_LABELS):
    ##                        # NN of DT/PRP$ JJ NN
    ##                        await self.control_plane.link_nouns_via_preposition(
    ##                            idx_to_noun_joined_base_form[idx-1],
    ##                            token,
    ##                            idx_to_noun_joined_base_form[idx+3])
    ##            elif 1 < idx:
    ##                if idx < token_cnt-1:
    ##                    if (ud_token_labels["xpos"][idx-2] in NOUN_LABELS
    ##                            and ud_token_labels["xpos"][idx-1] in STATE_OF_BEING_VERBS
    ##                            and ud_token_labels["xpos"][idx+1] in NOUN_LABELS):
    ##                        # NN is of NN
    ##                        await self.control_plane.link_nouns_via_preposition(
    ##                            idx_to_noun_joined_base_form[idx-2],
    ##                            token,
    ##                            idx_to_noun_joined_base_form[idx+1])
    ##                elif idx < token_cnt-2:
    ##                    if (ud_token_labels["xpos"][idx-2] in NOUN_LABELS
    ##                            and ud_token_labels["xpos"][idx-1] in STATE_OF_BEING_VERBS
    ##                            and ud_token_labels["xpos"][idx+1] in {"DT", "JJ", "PRP$"}
    ##                            and ud_token_labels["xpos"][idx+2] in NOUN_LABELS):
    ##                        # NN is of DT/JJ/PRP$ NN
    ##                        await self.control_plane.link_nouns_via_preposition(
    ##                            idx_to_noun_joined_base_form[idx-2],
    ##                            token,
    ##                            idx_to_noun_joined_base_form[idx+2])
    ##                elif idx < token_cnt-3:
    ##                    if (ud_token_labels["xpos"][idx-2] in NOUN_LABELS
    ##                            and ud_token_labels["xpos"][idx-1] in STATE_OF_BEING_VERBS
    ##                            and ud_token_labels["xpos"][idx+1] in {"DT", "PRP$"}
    ##                            and ud_token_labels["xpos"][idx+2] == "JJ"
    ##                            and ud_token_labels["xpos"][idx+3] in NOUN_LABELS):
    ##                        # NN is of DT/JJ/PRP$ NN
    ##                        await self.control_plane.link_nouns_via_preposition(
    ##                            idx_to_noun_joined_base_form[idx-2],
    ##                            token,
    ##                            idx_to_noun_joined_base_form[idx+3])

    ##    # Verbs
    ##    elif ud_xpos_label in VERB_LABELS:

    ##        if lowered_token in STATE_OF_BEING_VERBS:
    ##            if idx == 0 and idx < token_cnt - 1:
    ##                logger.info(f"state-of-being verb: {token} {ud_token_labels['tokens'][idx + 1]}")
    ##            elif idx < token_cnt - 1:
    ##                logger.info(
    ##                    f"state-of-being verb: {ud_token_labels['tokens'][idx - 1]} {token} {ud_token_labels['tokens'][idx + 1]}")

    ##        if deprel_label == "root":
    ##            last_root_verb = lowered_token
    ##            last_root_verb_idx = idx
    ##        last_verb = lowered_token
    ##        last_verb_idx = idx

    ##    #    # TODO:
    ##    #    # - When nouns are common, verbs often describe a capability of that kind of noun
    ##    #    #   - Verb nodes connect to nouns as COULD/_NOT and CAN/_NOT to indicate capability
    ##    #    #   - Verb nodes should store information about associated verb links
    ##    #    # - When nouns are proper, Verbs often describe actions individual do to each other
    ##    #    #   - Verb links connect individual nodes

    ##    if graphed_nsubj_root_obj_link is False and (last_root_verb and last_nsubj_noun and last_obj_noun):
    ##        await self.control_plane.link_nouns_via_root_verb(
    ##            last_nsubj_noun, wordnet_lemmatizer.lemmatize(last_root_verb, pos="v"), last_obj_noun)
    ##        graphed_nsubj_root_obj_link = True

    ##    last_walked_idx = idx
    pass