"""The human-meaningful and adversarial half of the gold set, written against real passages."""

from __future__ import annotations

from typing import Any

# --- 1.

HUMAN: list[dict[str, Any]] = [
    # --- exact term: a distinctive phrase the keyword leg should find -----------------------------
    dict(
        q="Which system at NATS failed and caused all the flight cancellations?",
        cat="exact_term", crit="major",
        ev=[("B9F5B681-6DD6-4F44-A68E-2580685A4A1D", "Swanwick flight processing system")],
        nug=[("The failure was in NATS's Swanwick flight processing system.", True),
             ("A cyber-attack had been ruled out.", False)],
    ),
    dict(
        q="What did the minister say the Scottish courts did to the Rosebank consents?",
        cat="exact_term", crit="major",
        ev=[("481BCEF7-EE75-46EC-BBD0-F9F2522C2D61", "Court of Session quashed the original consents")],
        nug=[("The Court of Session quashed the original consents in January 2025.", True)],
    ),
    dict(
        q="Who said the Falkland Islands will remain British for as long as the islanders wish?",
        cat="exact_term", crit="major",
        ev=[("6BAC9B38-6A3B-4229-98E0-DF49B27E47E8", "will remain British for as long as the Falkland Islanders")],
        nug=[("Wendy Morton said it, for the Conservative benches.", True)],
    ),
    dict(
        q="what's the Royal Mint Court thing about",
        cat="exact_term", crit="minor",
        ev=[("11037035-3333-497E-865D-97AC5887FC35", "new Chinese embassy at Royal Mint Court")],
        nug=[("It is the site proposed for a new Chinese embassy.", True),
             ("The decision sits with Planning Ministers, independently of the rest of Government.", False)],
        note="Deliberately lower-case and unpunctuated, the way a search box receives it.",
    ),
    dict(
        q="Which chalk stream did the MP for Hertford and Stortford talk about in the water debate?",
        cat="exact_term", crit="minor",
        ev=[("8F466684-7ADB-4C6D-8EE2-9BB9909122A4", "River Stort")],
        nug=[("The River Stort.", True)],
    ),
    dict(
        q="Somebody quoted Margaret Thatcher on not walking into other people's countries - who was it?",
        cat="exact_term", crit="major",
        ev=[("A0ED5380-BE8C-4738-BE9A-A881A55C4FF1", "we do not use it to walk into other people")],
        nug=[("Ed Davey quoted it, in the debate on Venezuela.", True)],
        note="Mention-vs-speaker done honestly: the answer is the quoter, and the question asks for him.",
    ),
    dict(
        q="What percentage cap did the government put on high-risk vendors in the 5G network?",
        cat="exact_term", crit="major",
        ev=[("A8C5E355-53FB-4168-B969-857DFD4E3EC6", "limited to a minority presence of up to 35%")],
        nug=[("A minority presence of up to 35% outside security-critical functions.", True)],
    ),
    dict(
        q="Which pub-and-restaurant sector did the Tories call the cornerstone of the economy in the seasonal work debate?",
        cat="exact_term", crit="minor",
        ev=[("10B58B20-A03C-4DCE-B90B-34CD45374319", "hospitality sector is the cornerstone")],
        nug=[("Hospitality.", True)],
    ),

    # --- paraphrase: almost no shared wording, so only the dense leg can bridge it ----------------
    dict(
        q="Is charging visitors for staying overnight going to hurt hotels and pubs?",
        cat="paraphrase", crit="major",
        ev=[("5C8CFAA5-F430-49A7-989B-7934BD052D36", "If we tax something, we get less of it")],
        nug=[("Rebecca Paul argued taxing tourism will produce less tourism.", True),
             ("She said hospitality had already lost almost 89,000 jobs.", False)],
    ),
    dict(
        q="Did anyone warn about drinking water being polluted by underground gas extraction?",
        cat="paraphrase", crit="major",
        ev=[("4D4A4F2B-159C-4346-9F48-0265841985A6", "groundwater contamination")],
        nug=[("Claire Young raised the potential for groundwater contamination.", True)],
    ),
    dict(
        q="Was there any argument that households cannot change supplier if they are unhappy?",
        cat="paraphrase", crit="minor",
        ev=[("C0D2A8C8-52D0-4B57-A3DD-5AE554DE52F2", "cannot suddenly decide that they have had enough")],
        nug=[("Sarah Hall made that argument for public ownership of water.", True)],
    ),
    dict(
        q="Did a peer say this was the worst schools announcement they had heard in decades in the chamber?",
        cat="paraphrase", crit="major",
        ev=[("BC6D4D31-DF50-45DE-AE11-5119636C71A9", "by far the worst education Statement")],
        nug=[("Baroness Blackstone, of the 33 years she had sat in the House.", True)],
    ),
    dict(
        q="Someone said firms with no income coming in had no option but to let staff go straight away.",
        cat="paraphrase", crit="major",
        ev=[("B622BCD6-E9D9-4AFF-87F9-980383CDE7C3", "no choice but to lay off workers now")],
        nug=[("Greg Clark, describing businesses facing a crisis.", True)],
    ),
    dict(
        q="Did anybody argue the poor get blamed for their circumstances while the rich credit themselves?",
        cat="paraphrase", crit="major",
        ev=[("070AC1AD-758B-4DDA-9BE4-55530270027B", "born into great wealth consider their position")],
        nug=[("Lizzi Collinge made that argument.", True)],
    ),
    dict(
        q="Who argued that the money for welfare ultimately comes out of somebody's pocket, now or later?",
        cat="paraphrase", crit="minor",
        ev=[("174220A0-6F8D-441F-9DA2-5ABC77CC86B4", "taxpayers of today")],
        nug=[("Sir Ashley Fox.", True)],
    ),
    dict(
        q="Was concern raised about self-employed people and freelancers being left out of the rescue package?",
        cat="paraphrase", crit="major",
        ev=[("9AA5C7B9-3E23-4F35-9248-50684A98B374", "sole traders, freelancers and the self-employed")],
        nug=[("David Linden raised it.", True)],
    ),
    dict(
        q="Did a peer bring up a farmer they met while talking about whether reforms can actually be paid for?",
        cat="paraphrase", crit="minor",
        ev=[("CB2EA721-7319-497E-A185-9B24EC577ECB", "I went to visit a pig farmer")],
        nug=[("Lord Sentamu, in the Sentencing Bill Committee.", True)],
    ),
    dict(
        q="Was there a complaint that first-time buyers are stuck renting while older owners cannot move out?",
        cat="paraphrase", crit="minor",
        ev=[("AFDCFA87-E050-486A-9B3C-5FD0B52E53AF", "stuck at the gatehouse")],
        nug=[("Sarah Bool, in the stamp duty debate.", True)],
    ),

    # --- numeric ----------------------------------------------------------------------------------
    dict(
        q="How many flights got cancelled in the air traffic control meltdown?",
        cat="numeric", crit="major",
        ev=[("66D8DADC-061C-4602-AFD9-4F4AB0E0B6B3", "flights cancelled")],
        nug=[("Over 2,000 flights, affecting more than 150,000 passengers.", True)],
        forbid=["Any figure presented as the Government's own count; this is the shadow Minister's."],
    ),
    dict(
        q="How much did the OBR say digital ID would cost?",
        cat="numeric", crit="major",
        ev=[("5003558C-5263-4847-BB68-7D86CF12814D", "reckons the scheme will cost")],
        nug=[("1.8 billion pounds, as cited by Julia Lopez.", True)],
        forbid=["Stating the figure as the Government's estimate rather than the OBR's, as quoted by a member."],
    ),
    dict(
        q="How many jobs went when Flybe collapsed?",
        cat="numeric", crit="major",
        ev=[("926894A8-90EB-472E-8728-8FC60D0CEC61", "2,000 jobs")],
        nug=[("About 2,000 direct jobs.", True)],
    ),
    dict(
        q="How many Flybe staff were based in Scotland?",
        cat="numeric", crit="minor",
        ev=[("DD73885F-5BF3-4455-93FB-230FAE23B375", "about 300 were based in Scotland")],
        nug=[("About 300 of the 2,000, with 130 in Gavin Newlands's constituency.", True)],
    ),
    dict(
        q="how many hospitality jobs have been lost according to the tories",
        cat="numeric", crit="minor",
        ev=[("5C8CFAA5-F430-49A7-989B-7934BD052D36", "hospitality jobs have gone")],
        nug=[("Almost 89,000, as claimed by Rebecca Paul.", True)],
        forbid=["Presenting the figure as an official statistic rather than as a claim made in debate."],
    ),
    dict(
        q="How many youth hubs are being set up and where?",
        cat="numeric", crit="minor",
        ev=[("0D69B72B-DE38-4818-9A24-CF3D36286E38", "youth hubs in 360 locations")],
        nug=[("360 locations across the country.", True)],
    ),
    dict(
        q="How many people signed the petition on local government reorganisation in Devon?",
        cat="numeric", crit="minor",
        ev=[("9205754F-79EA-4B42-8D46-CB0BF0A34F11", "4,200 signatures")],
        nug=[("More than 4,200, on a similar online petition.", True)],
    ),
    dict(
        q="How many of Liz Jarvis's constituents signed the digital ID petition?",
        cat="numeric", crit="minor",
        ev=[("3483FB35-5790-46E7-8128-DF25526A7A53", "4,000 people from my constituency")],
        nug=[("4,000.", True)],
    ),

    # --- named speaker: the name is not in the sentence text, so plain keyword search cannot use it -
    dict(
        q="What did Rachel Reeves say about why the black hole was not known about earlier?",
        cat="named_speaker", crit="major",
        ev=[("8DA64417-CE8A-407A-8B07-69EBB58DD022", "did not tell the OBR these numbers")],
        nug=[("She said the Conservative party did not give the OBR the numbers.", True)],
        forbid=["Asserting the size of the black hole as an established fact; it is a contested claim."],
    ),
    dict(
        q="What did Tulip Siddiq say about her own family when talking about Palestinian visas?",
        cat="named_speaker", crit="major",
        ev=[("1EEFAA24-F575-4D73-A002-7B73D7B08B3B", "My mother came to this country")],
        nug=[("That her mother came to the UK as a political asylum seeker in the 1970s.", True)],
    ),
    dict(
        q="What point did Dan Jarvis make about the intelligence agencies in the Chinese embassy row?",
        cat="named_speaker", crit="major",
        ev=[("F63858DA-8F69-45E7-BD87-D8052ABC4062", "director general of MI5")],
        nug=[("He urged members to look at what the heads of MI5 and GCHQ had said.", True)],
    ),
    dict(
        q="What did Baroness Hoey say the Windsor Framework meant for the vapes bill?",
        cat="named_speaker", crit="major",
        ev=[("F75BB865-C1D4-4F99-B230-279CDC4575D5", "When the Windsor Framework was proposed")],
        nug=[("She recalled that she and others had warned about it when it was proposed.", True)],
    ),
    dict(
        q="What did Helen Morgan object to in the medical training bill?",
        cat="named_speaker", crit="major",
        ev=[("BEAAFC61-67A2-4E33-8816-DE82068D028B", "negative procedure")],
        nug=[("That clause 7(1) would let Ministers change eligibility by the negative procedure.", True),
             ("The Liberal Democrats supported the Bill overall.", False)],
    ),
    dict(
        q="what did Caroline Nokes say about getting time for the womens day debate",
        cat="named_speaker", crit="minor",
        ev=[("C3A96E78-75A3-4D1E-937B-ECB1C0560B6D", "Backbench Business Committee")],
        nug=[("She thanked the Leader of the House for Government time, the debate normally coming from the "
              "Backbench Business Committee.", True)],
    ),
    dict(
        q="What did Sir Peter Bottomley say about how long he had known people infected?",
        cat="named_speaker", crit="major",
        ev=[("AB486DDB-7316-43AD-ACF2-AA1EBA614EEA", "36 years since I was with the first of my friends")],
        nug=[("36 years since the first friend he knew to have been infected; 33 since that person died.", True)],
    ),
    dict(
        q="What did Baroness Fox of Buckley welcome in the tobacco bill?",
        cat="named_speaker", crit="minor",
        ev=[("593D5571-05CD-407B-8A54-106E27301E50", "vape vending machines in mental health hospitals")],
        nug=[("Government amendments exempting vape vending machines in mental health hospitals.", True)],
    ),

    # --- office at a date: the question names a role, not a person -------------------------
    dict(
        q="What did the policing minister say about the Villa Park away fans ban?",
        cat="office_at_date", crit="major", behaviour="answer",
        ev=[("5EAC4AA8-6B27-4068-989D-0ABC3BC272D8", "ban away fans at Villa Park")],
        nug=[("Sarah Jones, the Minister for Policing and Crime, acknowledged the concern and updated the House.",
              True)],
        note="The role has to resolve to a person at that date before the answer means anything.",
    ),
    dict(
        q="What did the housing minister say about who decides the Chinese embassy planning application?",
        cat="office_at_date", crit="major", behaviour="answer",
        ev=[("11037035-3333-497E-865D-97AC5887FC35", "decision to be taken by Planning Ministers")],
        nug=[("Matthew Pennycook said it is for Planning Ministers, independent of the rest of Government.", True)],
    ),
    dict(
        q="What did the climate minister say about the Finch judgment and end-use emissions?",
        cat="office_at_date", crit="major", behaviour="answer",
        ev=[("481BCEF7-EE75-46EC-BBD0-F9F2522C2D61", "Finch judgment")],
        nug=[("Katie White said supplementary guidance implements Finch, so end-use combustion emissions "
              "must be assessed.", True)],
    ),
    dict(
        q="What did the DWP minister say when winding up the WASPI ombudsman debate?",
        cat="office_at_date", crit="major", behaviour="answer",
        ev=[("8215A6AA-0577-4BED-B38B-FF764E7F24A4", "Backbench Business Committee for finding the time")],
        nug=[("Paul Maynard, the Parliamentary Under-Secretary for Work and Pensions, replied for the "
              "Government.", True)],
    ),
    dict(
        q="Which minister moved the government amendment in the seasonal work debate?",
        cat="office_at_date", crit="minor", behaviour="answer",
        ev=[("C9B38CA8-5ABE-49E9-903D-F9C710AA15A0", "I beg to move an amendment")],
        nug=[("Blair McDougall, Parliamentary Under-Secretary for Business and Trade.", True)],
    ),

    # --- party attribution ------------------------------------------------------------------------
    dict(
        q="Which party was attacking the visitor levy as a holiday tax?",
        cat="party_attribution", crit="major",
        ev=[("5C8CFAA5-F430-49A7-989B-7934BD052D36", "holiday tax")],
        nug=[("The Conservatives; Rebecca Paul used the phrase.", True)],
        forbid=["Attributing the phrase to the Government, who published the levy."],
    ),
    dict(
        q="Which party opposed the employer national insurance rise in the high streets debate?",
        cat="party_attribution", crit="major",
        ev=[("AB981B23-E186-4FD3-B273-F3AC9B04E5F6", "employer national insurance increase")],
        nug=[("The Liberal Democrats; Adam Dance spoke for them.", True)],
    ),
    dict(
        q="Was it an SNP member who raised Scottish job losses when Flybe went under?",
        cat="party_attribution", crit="minor",
        ev=[("DD73885F-5BF3-4455-93FB-230FAE23B375", "about 300 were based in Scotland")],
        nug=[("Yes: Gavin Newlands, SNP.", True)],
    ),
    dict(
        q="Which benches did the MP defending stamp duty abolition sit on?",
        cat="party_attribution", crit="minor",
        ev=[("5B961362-CA26-409D-8A75-AA1F4F797E67", "abolition of stamp duty land tax on primary residences")],
        nug=[("The Conservative benches; Sir Mel Stride moved the motion.", True)],
    ),
    dict(
        q="Was the peer who welcomed the vaping machine exemption a party peer or a crossbencher?",
        cat="party_attribution", crit="minor",
        ev=[("593D5571-05CD-407B-8A54-106E27301E50", "vape vending machines in mental health hospitals")],
        nug=[("Neither: Baroness Fox of Buckley sits as non-affiliated.", True)],
        note="The three-way answer is the point; 'crossbencher' and 'non-affiliated' are different things.",
    ),

    # --- multi-speech: the answer needs more than one contribution ---------------------------------
    dict(
        q="What were the arguments for and against the tourist tax?",
        cat="multi_speech", crit="major",
        ev=[("3955DBD7-51E4-4DE0-AFCD-191443AA268F", "empowering local leaders"),
            ("5C8CFAA5-F430-49A7-989B-7934BD052D36", "If we tax something, we get less of it")],
        nug=[("The Government framed it as empowering local leaders to invest.", True),
             ("The Opposition called it a holiday tax that would cost hospitality jobs.", True)],
    ),
    dict(
        q="What did both sides say about fracking and whether it is safe?",
        cat="multi_speech", crit="major",
        ev=[("9BEA6E54-028C-44C0-B169-57B97A76BF1E", "20 million people in the United States"),
            ("4D4A4F2B-159C-4346-9F48-0265841985A6", "groundwater contamination")],
        nug=[("Richard Tice pointed to millions living near US pads.", True),
             ("Claire Young raised groundwater contamination and the thinness of the evidence base.", True)],
    ),
    dict(
        q="How did MPs split on the Rosebank and Jackdaw consultation?",
        cat="multi_speech", crit="major",
        ev=[("481BCEF7-EE75-46EC-BBD0-F9F2522C2D61", "Court of Session quashed the original consents"),
            ("BC00141D-044E-414E-9FD5-2B3D19C53628", "consultation on Rosebank and Jackdaw has now closed")],
        nug=[("The Minister set out the legal history and the Finch guidance.", True),
             ("Andrew Bowie said the consultation had closed and accused the Department of dithering.", True)],
    ),
    dict(
        q="What did peers say for and against the two-child limit being lifted?",
        cat="multi_speech", crit="major",
        ev=[("070AC1AD-758B-4DDA-9BE4-55530270027B", "born into great wealth consider their position"),
            ("174220A0-6F8D-441F-9DA2-5ABC77CC86B4", "safety net designed to support people in hardship")],
        nug=[("Lizzi Collinge argued about how poverty is attributed.", True),
             ("Sir Ashley Fox argued the cost falls on taxpayers now or later.", True)],
        forbid=["Describing this as a Lords debate; both speeches are from the Commons."],
        note="A scoping trap inside a multi-speech item: the question says 'peers' and the record says MPs.",
    ),
    dict(
        q="What concerns were raised about the Government's coronavirus support for workers?",
        cat="multi_speech", crit="major",
        ev=[("B622BCD6-E9D9-4AFF-87F9-980383CDE7C3", "no choice but to lay off workers now"),
            ("9AA5C7B9-3E23-4F35-9248-50684A98B374", "sole traders, freelancers and the self-employed")],
        nug=[("Greg Clark said firms were laying people off before help arrived.", True),
             ("David Linden said the self-employed were not covered.", True)],
    ),
    dict(
        q="What did both the Government and the opposition say about the air traffic failure?",
        cat="multi_speech", crit="major",
        ev=[("B9F5B681-6DD6-4F44-A68E-2580685A4A1D", "Swanwick flight processing system"),
            ("66D8DADC-061C-4602-AFD9-4F4AB0E0B6B3", "flights cancelled")],
        nug=[("The Minister identified the Swanwick flight processing system and ruled out a cyber-attack.", True),
             ("The shadow Minister put the disruption at over 2,000 flights and criticised the lack of a "
              "statement.", True)],
    ),

    # --- true multi-hop: speech -> speaker -> party or role ----------------------------------------
    dict(
        q="The MP who complained about 2,000 cancelled flights - which party do they belong to?",
        cat="multi_hop", crit="major",
        ev=[("66D8DADC-061C-4602-AFD9-4F4AB0E0B6B3", "flights cancelled")],
        nug=[("Greg Smith, Conservative.", True)],
    ),
    dict(
        q="Who moved the Government's amendment on seasonal work, and what department are they in?",
        cat="multi_hop", crit="major",
        ev=[("C9B38CA8-5ABE-49E9-903D-F9C710AA15A0", "I beg to move an amendment")],
        nug=[("Blair McDougall, at Business and Trade.", True)],
    ),
    dict(
        q="The peer who said this was the worst education statement in 33 years - what party do they take the whip for?",
        cat="multi_hop", crit="major",
        ev=[("BC6D4D31-DF50-45DE-AE11-5119636C71A9", "by far the worst education Statement")],
        nug=[("Baroness Blackstone, who sits as Independent Labour.", True)],
    ),
    dict(
        q="Who raised the Devon reorganisation petition and which party are they?",
        cat="multi_hop", crit="minor",
        ev=[("9205754F-79EA-4B42-8D46-CB0BF0A34F11", "4,200 signatures")],
        nug=[("Rebecca Smith, Conservative.", True)],
    ),
    dict(
        q="The member who quoted Thatcher about not invading other countries - what party leads them?",
        cat="multi_hop", crit="major",
        ev=[("A0ED5380-BE8C-4738-BE9A-A881A55C4FF1", "we do not use it to walk into other people")],
        nug=[("Ed Davey is himself the Liberal Democrat leader.", True)],
        note="The hop folds back on itself, which is where a naive chain answers with somebody else's name.",
    ),

    # --- temporal: first / last / when --------------------------------------------------------------
    dict(
        q="When did the government publish its response to the visitor levy consultation?",
        cat="temporal", crit="major",
        ev=[("3955DBD7-51E4-4DE0-AFCD-191443AA268F", "Last Thursday, the Government published their response")],
        nug=[("The Minister said 'last Thursday', speaking on 15 September 2026.", True)],
        forbid=["Converting 'last Thursday' to a date without saying the conversion was made."],
        note="A relative date in the record. Resolving it silently is how a plausible wrong date gets asserted.",
    ),
    dict(
        q="When were the Rosebank consents quashed?",
        cat="temporal", crit="major",
        ev=[("481BCEF7-EE75-46EC-BBD0-F9F2522C2D61", "January 2025")],
        nug=[("January 2025.", True)],
    ),
    dict(
        q="When was the away fans ban at Villa Park?",
        cat="temporal", crit="major",
        ev=[("5EAC4AA8-6B27-4068-989D-0ABC3BC272D8", "6 November")],
        nug=[("6 November.", True)],
    ),
    dict(
        q="When did Flybe actually collapse?",
        cat="temporal", crit="major",
        ev=[("DD73885F-5BF3-4455-93FB-230FAE23B375", "about 300 were based in Scotland")],
        nug=[("The Commons discussed it on 5 March 2020, as news of the collapse.", True)],
        forbid=["A date for the collapse itself that the record does not state."],
    ),
    dict(
        q="How long had Baroness Blackstone been in the Lords when she made that criticism?",
        cat="temporal", crit="minor",
        ev=[("BC6D4D31-DF50-45DE-AE11-5119636C71A9", "33 years I have sat in this House")],
        nug=[("33 years, by her own account.", True)],
    ),
    dict(
        q="Which came first in the record, the digital ID debate or the Maccabi away fans statement?",
        cat="temporal", crit="minor",
        ev=[("5003558C-5263-4847-BB68-7D86CF12814D", "reckons the scheme will cost"),
            ("5EAC4AA8-6B27-4068-989D-0ABC3BC272D8", "ban away fans at Villa Park")],
        nug=[("Both were on 8 December 2025.", True)],
        note="The trap is the assumption that two things must be orderable; here they are the same day.",
    ),

    # --- comparison ---------------------------------------------------------------------------------
    dict(
        q="Who was more critical of the government on jury trials, the Conservatives or Labour backbenchers?",
        cat="comparison", crit="major",
        ev=[("5C530750-9B00-4C1A-867D-1B7810720AC1", "reducing it, as the Government propose"),
            ("EED20239-B060-4931-9157-6134AACD4301", "justice delayed is justice denied")],
        nug=[("Both quoted speeches are Conservative; the comparison cannot be made from them alone.", True)],
        forbid=["A confident ranking of the two groups from these speeches."],
        note="A comparison the evidence does not support. The right answer names the limit.",
    ),
    dict(
        q="Was the reaction to the visitor levy harsher than the reaction to the stamp duty proposals?",
        cat="comparison", crit="minor",
        ev=[("5C8CFAA5-F430-49A7-989B-7934BD052D36", "If we tax something, we get less of it"),
            ("5B961362-CA26-409D-8A75-AA1F4F797E67", "abolition of stamp duty land tax on primary residences")],
        nug=[("They are different debates with different movers; any comparison is impressionistic.", True)],
        forbid=["Presenting a numeric or definitive comparison of 'harshness'."],
    ),
    dict(
        q="Compare what the Government and the Lib Dems said about the local government reorganisation plans.",
        cat="comparison", crit="major",
        ev=[("0DCF0CDE-3B23-472C-8F8D-EBB91BE6F593", "finally listened, but they have done so too late")],
        nug=[("Zöe Franklin said the Government had listened too late and the damage was done.", True)],
    ),
    dict(
        q="Did more MPs talk about hospitality jobs or about groundwater in September 2026?",
        cat="comparison", crit="minor",
        ev=[("5C8CFAA5-F430-49A7-989B-7934BD052D36", "hospitality jobs have gone")],
        nug=[("A count question, not a retrieval one; it needs the whole record, not a passage.", True)],
        forbid=["A count stated without saying it was computed over the whole window."],
    ),
    dict(
        q="Was the two-child limit debate more about money or about morality?",
        cat="comparison", crit="minor",
        ev=[("070AC1AD-758B-4DDA-9BE4-55530270027B", "born into great wealth consider their position"),
            ("174220A0-6F8D-441F-9DA2-5ABC77CC86B4", "safety net designed to support people in hardship")],
        nug=[("Both threads are present: one speech argues about attribution of blame, the other about cost.",
              True)],
    ),

    # --- stance under the three-case motion model ---------------------------------------------------
    dict(
        q="Does the government support public ownership of water?",
        cat="stance", crit="critical", behaviour="partial",
        ev=[("C0D2A8C8-52D0-4B57-A3DD-5AE554DE52F2", "I support public ownership of our water industry")],
        nug=[("The speech quoted is a backbencher's own position, not a Government position.", True)],
        forbid=["Presenting a backbench Labour member's support as the Government's policy."],
        note="Backbencher-for-Government-position is the most common attribution error in this corpus.",
    ),
    dict(
        q="Is the Conservative party in favour of the visitor levy?",
        cat="stance", crit="major", behaviour="answer",
        ev=[("5C8CFAA5-F430-49A7-989B-7934BD052D36", "Let us call this what it is")],
        nug=[("No: the Conservative speech attacks it as a holiday tax.", True)],
    ),
    dict(
        q="Do the Lib Dems back the medical training bill or not?",
        cat="stance", crit="major", behaviour="answer",
        ev=[("BEAAFC61-67A2-4E33-8816-DE82068D028B", "overall support for the Bill")],
        nug=[("They support the Bill overall while objecting to clause 7(1).", True)],
        forbid=["Reporting their amendments as opposition to the Bill."],
    ),
    dict(
        q="Was Baroness Hoey for or against the tobacco bill's Northern Ireland provisions?",
        cat="stance", crit="major", behaviour="answer",
        ev=[("F75BB865-C1D4-4F99-B230-279CDC4575D5", "When the Windsor Framework was proposed")],
        nug=[("Against, and she said she had warned about it at the time of the Windsor Framework.", True)],
    ),
    dict(
        q="Did Richard Burgon say he would vote for or against the government amendment on Lord Mandelson?",
        cat="stance", crit="major", behaviour="answer",
        ev=[("DA5DAE6C-B4AE-488D-B3AA-7247CD2869CF", "I will of course vote against it")],
        nug=[("Against, saying it would stop the full truth coming out.", True)],
    ),

    # --- house / section scoping --------------------------------------------------------------------
    dict(
        q="What did the Lords say about the leasehold bill and property agents?",
        cat="scoping", crit="major",
        ev=[("AF41CB09-0D7A-4BA1-A75C-44685A6CBAD9", "regulation of property agents")],
        nug=[("Lord Gascoigne responded to amendments on regulating property agents.", True)],
    ),
    dict(
        q="Was the digital ID debate in the main chamber or Westminster Hall?",
        cat="scoping", crit="major",
        ev=[("3483FB35-5790-46E7-8128-DF25526A7A53", "e-petition")],
        nug=[("Westminster Hall, as an e-petition debate.", True)],
    ),
    dict(
        q="Which house debated deferment rates in the leasehold bill?",
        cat="scoping", crit="minor",
        ev=[("3F0E2741-5FB8-448D-8EA2-2D595F128F51", "deferment rates")],
        nug=[("The Lords.", True)],
    ),
    dict(
        q="Did peers or MPs debate the Huawei 35% cap?",
        cat="scoping", crit="major",
        ev=[("A8C5E355-53FB-4168-B969-857DFD4E3EC6", "limited to a minority presence of up to 35%")],
        nug=[("MPs, in Westminster Hall.", True)],
    ),
    dict(
        q="What was said in Westminster Hall about water company ownership?",
        cat="scoping", crit="major",
        ev=[("8F466684-7ADB-4C6D-8EE2-9BB9909122A4", "River Stort"),
            ("C0D2A8C8-52D0-4B57-A3DD-5AE554DE52F2", "I support public ownership of our water industry")],
        nug=[("Both speeches quoted are from the Westminster Hall debate on the water sector.", True)],
    ),
    dict(
        q="Was the youth hubs answer given in the Commons?",
        cat="scoping", crit="major",
        ev=[("0D69B72B-DE38-4818-9A24-CF3D36286E38", "youth hubs in 360 locations")],
        nug=[("No: Baroness Sherlock said it in the Lords.", True)],
        forbid=["Placing this answer in the Commons."],
    ),

    # --- quote verification, including negation flips -----------------------------------------------
    dict(
        q="Did someone say in the visitor levy debate that if you tax something you get less of it?",
        cat="quote_verification", crit="major", behaviour="answer",
        ev=[("5C8CFAA5-F430-49A7-989B-7934BD052D36", "If we tax something, we get less of it")],
        nug=[("Yes, Rebecca Paul said it.", True)],
    ),
    dict(
        q="Did the minister say a cyber-attack caused the air traffic disruption?",
        cat="quote_verification", crit="critical", behaviour="correct_false_premise",
        ev=[("B9F5B681-6DD6-4F44-A68E-2580685A4A1D", "cyber-attack has now been ruled out")],
        nug=[("No: he said a cyber-attack had been ruled out and blamed a technical issue.", True)],
        forbid=["That the Minister attributed the disruption to a cyber-attack."],
        note="Negation flip: the passage contains the words, with the opposite polarity.",
    ),
    dict(
        q="Did Claire Young say the evidence base showed lower-volume fracking was safe?",
        cat="quote_verification", crit="critical", behaviour="correct_false_premise",
        ev=[("4D4A4F2B-159C-4346-9F48-0265841985A6", "the evidence base is not there")],
        nug=[("No: she said Ministers claim the evidence base is not there, and pressed them on it.", True)],
        forbid=["That she asserted lower-volume fracking is safe."],
    ),
    dict(
        q="Is it right that Sir Ashley Fox said benefits are not paid for by taxpayers?",
        cat="quote_verification", crit="critical", behaviour="correct_false_premise",
        ev=[("174220A0-6F8D-441F-9DA2-5ABC77CC86B4", "taxpayers of today")],
        nug=[("No, the opposite: he said they are paid by taxpayers today or tomorrow.", True)],
        forbid=["That he denied benefits are funded by taxpayers."],
    ),
    dict(
        q="Did anyone actually use the words 'holiday tax' in the Commons?",
        cat="quote_verification", crit="major", behaviour="answer",
        ev=[("5C8CFAA5-F430-49A7-989B-7934BD052D36", "holiday tax")],
        nug=[("Yes: Rebecca Paul, on 15 September 2026.", True)],
    ),
    dict(
        q="Did Greg Clark say businesses had plenty of time before laying people off?",
        cat="quote_verification", crit="critical", behaviour="correct_false_premise",
        ev=[("B622BCD6-E9D9-4AFF-87F9-980383CDE7C3", "no choice but to lay off workers now")],
        nug=[("No: he said they saw no choice but to lay off workers now.", True)],
        forbid=["That he said businesses were not under immediate pressure."],
    ),

    # --- colloquial and misspelt: the register the maintainer asked for ----------------------------------
    dict(
        q="whats this tourist tax thing labour are bringing in",
        cat="colloquial", crit="major", behaviour="answer",
        ev=[("3955DBD7-51E4-4DE0-AFCD-191443AA268F", "overnight visitor levy")],
        nug=[("It is the overnight visitor levy; the Government published its consultation response in "
              "September 2026.", True),
             ("Opponents call it a holiday tax.", False)],
        forbid=["Adopting 'tourist tax' as the record's own term without saying the record says "
                "'overnight visitor levy'."],
        note="Popular label vs the record's term, with no capitals and no question mark.",
    ),
    dict(
        q="did the tories U-turn on fracking",
        cat="colloquial", crit="major", behaviour="answer",
        ev=[("9BEA6E54-028C-44C0-B169-57B97A76BF1E", "20 million people in the United States")],
        nug=[("The record shows Richard Tice arguing for fracking, not a Conservative reversal.", True)],
        forbid=["Asserting a U-turn the record does not show."],
    ),
    dict(
        q="whos the mp thats always banging on about chalk streams",
        cat="colloquial", crit="minor", behaviour="answer",
        ev=[("8F466684-7ADB-4C6D-8EE2-9BB9909122A4", "River Stort")],
        nug=[("Josh Dean raised the River Stort in the water sector debate.", True)],
        forbid=["Characterising a member as 'always' doing something on the strength of one speech."],
    ),
    dict(
        q="is the goverment scrapping the two child benifit cap or not",
        cat="colloquial", crit="major", behaviour="answer",
        ev=[("070AC1AD-758B-4DDA-9BE4-55530270027B", "children live in households unable to afford")],
        nug=[("A private Member's Bill to remove the limit was debated on 3 February 2026.", True)],
        forbid=["Stating that the Government has abolished it."],
        note="Two misspellings ('goverment', 'benifit') on purpose.",
    ),
    dict(
        q="why do we need id cards now, whats the point",
        cat="colloquial", crit="major", behaviour="answer",
        ev=[("3483FB35-5790-46E7-8128-DF25526A7A53", "mandatory digital ID")],
        nug=[("Constituents petitioned against mandatory digital ID; the debate was on 8 December 2025.", True)],
        forbid=["Treating the petitioners' objections as the Government's own case."],
    ),
    dict(
        q="Starmer government vs the lords on vapes whats going on",
        cat="colloquial", crit="minor", behaviour="answer",
        ev=[("593D5571-05CD-407B-8A54-106E27301E50", "vape vending machines in mental health hospitals")],
        nug=[("Peers debated Report stage amendments; the Government moved an exemption for vape vending "
              "machines in mental health hospitals.", True)],
    ),
    dict(
        q="did they kick the chinese embassy application out",
        cat="colloquial", crit="major", behaviour="answer",
        ev=[("11037035-3333-497E-865D-97AC5887FC35", "decision to be taken by Planning Ministers")],
        nug=[("The record has the Minister saying it is for Planning Ministers to decide, not that it was "
              "refused.", True)],
        forbid=["Stating that the application was rejected."],
    ),
    dict(
        q="my nan lost out on her pension, did anyone in parliament do anything",
        cat="colloquial", crit="major", behaviour="answer",
        ev=[("3F103EDB-7B3B-442F-8866-E1FC1C42D95C", "compensation scheme should look like")],
        nug=[("The Commons debated the ombudsman's report on women's state pension age on 16 May 2024.", True)],
        forbid=["Promising an outcome or entitlement to the questioner."],
        note="A constituent, not a researcher. The right answer is factual and does not give advice.",
    ),
    dict(
        q="whens the assisted dying bill coming back",
        cat="colloquial", crit="major", behaviour="answer",
        ev=[("EDF06F9D-F9C1-456F-A666-1461032275E8", "most vulnerable and least able to advocate")],
        nug=[("The Terminally Ill Adults (End of Life) Bill was debated on 11 September 2026.", True)],
        forbid=["A future date the record does not contain."],
    ),
    dict(
        q="how much are we paying for the chagos deal",
        cat="colloquial", crit="critical", behaviour="partial",
        ev=[("5003558C-5263-4847-BB68-7D86CF12814D", "Chagos islands")],
        nug=[("The passage mentions the Chagos islands only in passing, as part of a spending argument.", True)],
        forbid=["Quoting a Chagos cost figure drawn from a passage that is about digital ID."],
        note="Near-miss: the words are present, the answer is not.",
    ),
    dict(
        q="Boris and the coronavirus bill - what did he actually promise",
        cat="colloquial", crit="major", behaviour="answer",
        ev=[("B622BCD6-E9D9-4AFF-87F9-980383CDE7C3", "no choice but to lay off workers now")],
        nug=[("The retrieved debates are on employment support; the quoted speech is Greg Clark's, not the "
              "Prime Minister's.", True)],
        forbid=["Attributing a backbench or Government speech to Boris Johnson without evidence."],
    ),
    dict(
        q="did that bloke really compare the housing market to a castle lol",
        cat="colloquial", crit="minor", behaviour="answer",
        ev=[("AFDCFA87-E050-486A-9B3C-5FD0B52E53AF", "castle")],
        nug=[("Sarah Bool did, in the stamp duty debate; the member is a woman.", True)],
        forbid=["Referring to the speaker as 'he'."],
        note="A wrong-gender assumption in the question. Repeating it is a small harm that is easy to miss.",
    ),

    # --- multi-part ----------------------------------------------------------------------------------
    dict(
        q="Who ran the water debate in Westminster Hall, what did they want, and did the Government agree?",
        cat="multi_part", crit="major", behaviour="partial",
        ev=[("C0D2A8C8-52D0-4B57-A3DD-5AE554DE52F2", "I support public ownership of our water industry")],
        nug=[("Members including Sarah Hall argued for public ownership.", True),
             ("The quoted passages do not contain a Government response, so that part is unanswered.", True)],
        note="Explicitly partial: a system that answers all three parts has invented the third.",
    ),
    dict(
        q="What is the Villa Park ban, who decided it, and has it been reversed?",
        cat="multi_part", crit="major", behaviour="partial",
        ev=[("5EAC4AA8-6B27-4068-989D-0ABC3BC272D8", "ban away fans at Villa Park")],
        nug=[("Away fans were banned at Villa Park on 6 November.", True),
             ("The Minister described police involvement and ongoing developments.", False),
             ("Whether it was reversed is not established by the quoted record.", True)],
    ),
    dict(
        q="Tell me the cost of digital ID, who says so, and whether the Government accepts it.",
        cat="multi_part", crit="major", behaviour="partial",
        ev=[("5003558C-5263-4847-BB68-7D86CF12814D", "reckons the scheme will cost")],
        nug=[("1.8 billion pounds.", True),
             ("The OBR, as cited by Julia Lopez.", True),
             ("The passage does not record the Government accepting the figure.", True)],
    ),
    dict(
        q="Who spoke about infected blood, what did they say, and what did the Government offer?",
        cat="multi_part", crit="major", behaviour="partial",
        ev=[("DDB1530A-91D0-4171-AE23-27A6B8C3DDBD", "standing ovation")],
        nug=[("Chris Stephens spoke, with constituents present.", True),
             ("The quoted passage does not set out the Government's offer.", True)],
    ),
    dict(
        q="What is the 35% Huawei cap, who announced it, and is it still in force?",
        cat="multi_part", crit="major", behaviour="partial",
        ev=[("A8C5E355-53FB-4168-B969-857DFD4E3EC6", "limited to a minority presence of up to 35%")],
        nug=[("High-risk vendors limited to a minority presence of up to 35% outside security-critical "
              "functions.", True),
             ("Matt Warman set it out in March 2020.", True),
             ("Whether it remains in force is outside the record's dates.", True)],
    ),

    # --- cross-window: the same subject in two different windows -------------------------------------
    dict(
        q="Has local government reorganisation come up more than once in the record?",
        cat="cross_window", crit="major",
        ev=[("0DCF0CDE-3B23-472C-8F8D-EBB91BE6F593", "finally listened, but they have done so too late")],
        nug=[("Yes: it is debated in September 2026 and again in early 2026.", True)],
    ),
    dict(
        q="Was the Diego Garcia base debated in more than one session?",
        cat="cross_window", crit="minor",
        ev=[("5003558C-5263-4847-BB68-7D86CF12814D", "Chagos islands")],
        nug=[("The Bill appears in both late 2025 and early 2026 in the record.", True)],
        forbid=["A claim about sessions the record's windows cannot support."],
    ),
    dict(
        q="Did the Terminally Ill Adults Bill appear in more than one part of the record?",
        cat="cross_window", crit="major",
        ev=[("EDF06F9D-F9C1-456F-A666-1461032275E8", "most vulnerable and least able to advocate")],
        nug=[("Yes: it appears in late 2025, early 2026 and September 2026.", True)],
    ),
    dict(
        q="Was the Crime and Policing Bill still going through in 2026?",
        cat="cross_window", crit="minor",
        ev=[("04A5D1DE-6951-4AAC-BC40-43F9733A5DC3", "violence against women and girls")],
        nug=[("The Sentencing Bill was in Lords Committee in December 2025; Crime and Policing runs into "
              "2026.", True)],
    ),
    dict(
        q="Has the record got anything on the Armed Forces Bill from more than one period?",
        cat="cross_window", crit="minor",
        ev=[("6BAC9B38-6A3B-4229-98E0-DF49B27E47E8", "British soldiers fought and died")],
        nug=[("The Armed Forces Bill appears in both early and September 2026.", True)],
        forbid=["Citing the Falklands speech as an Armed Forces Bill speech."],
    ),

    # --- boilerplate traps ---------------------------------------------------------------------------
    dict(
        q="Who said it is a pleasure to serve under your chairmanship?",
        cat="boilerplate_trap", crit="minor", behaviour="clarify_ambiguous",
        ev=[],
        nug=[("The phrase is procedural and used by many members; the question needs narrowing.", True)],
        forbid=["Naming one member as though the phrase identified them."],
        prov="adversarial",
    ),
    dict(
        q="What did the Speaker say about the sitting being suspended?",
        cat="boilerplate_trap", crit="minor", behaviour="clarify_ambiguous",
        ev=[],
        nug=[("Suspension formulae are procedural boilerplate and are excluded from the searchable text.",
              True)],
        forbid=["Quoting a substantive speech as though it were the suspension announcement."],
        prov="adversarial",
    ),
    dict(
        q="Find the bit where they say the question is that the amendment be made",
        cat="boilerplate_trap", crit="minor", behaviour="clarify_ambiguous",
        ev=[],
        nug=[("That is a standard question put by the Chair, recorded throughout the corpus.", True)],
        forbid=["Presenting one instance as the only one."],
        prov="adversarial",
    ),
]
