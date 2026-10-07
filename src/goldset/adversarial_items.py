"""The adversarial half: items written to be failed, by a system and by a careless human alike."""

from __future__ import annotations

from typing import Any

ADVERSARIAL: list[dict[str, Any]] = [
    # --- misattribution: the right words, the wrong mouth -------------------------------------------
    dict(
        q="What did Kemi Badenoch say about the Swanwick flight processing system?",
        cat="misattribution", crit="critical", prov="adversarial",
        ev=[("B9F5B681-6DD6-4F44-A68E-2580685A4A1D", "Swanwick flight processing system")],
        nug=[("She did not: the Swanwick explanation was given by Keir Mather for the Government.", True)],
        forbid=["Any quotation about Swanwick attributed to Kemi Badenoch."],
    ),
    dict(
        q="Why did Rachel Reeves call the visitor levy a holiday tax?",
        cat="misattribution", crit="critical", prov="adversarial",
        ev=[("5C8CFAA5-F430-49A7-989B-7934BD052D36", "holiday tax")],
        nug=[("She did not: the phrase is Rebecca Paul's, from the Conservative benches.", True)],
        forbid=["Attributing 'holiday tax' to Rachel Reeves or to any Government member."],
    ),
    dict(
        q="What did Lord Moylan say about regulating property agents?",
        cat="misattribution", crit="major", prov="adversarial",
        ev=[("AF41CB09-0D7A-4BA1-A75C-44685A6CBAD9", "regulation of property agents")],
        nug=[("Property agents were answered by Lord Gascoigne; Lord Moylan spoke to deferment rates.", True)],
        forbid=["A property-agents position attributed to Lord Moylan."],
        note="Both peers are Conservative and both spoke in the same group, which is what makes it stick.",
    ),
    dict(
        q="What did Ed Davey say about the assessments of MI5 and GCHQ on the Chinese embassy?",
        cat="misattribution", crit="critical", prov="adversarial",
        ev=[("F63858DA-8F69-45E7-BD87-D8052ABC4062", "director general of MI5")],
        nug=[("That was Dan Jarvis, for the Government, not Ed Davey.", True)],
        forbid=["An MI5 or GCHQ point attributed to Ed Davey."],
    ),
    dict(
        q="What did Sarah Jones say about the Chinese embassy planning decision?",
        cat="misattribution", crit="major", prov="adversarial",
        ev=[("11037035-3333-497E-865D-97AC5887FC35", "decision to be taken by Planning Ministers")],
        nug=[("The planning answer came from Matthew Pennycook; Sarah Jones answered on the away fans ban.", True)],
        forbid=["A planning statement attributed to Sarah Jones."],
        note="Two Ministers, two urgent questions in the same window. Plausible, and wrong.",
    ),
    dict(
        q="What did the Chancellor say about the number of Flybe jobs lost in Scotland?",
        cat="misattribution", crit="major", prov="adversarial",
        ev=[("DD73885F-5BF3-4455-93FB-230FAE23B375", "about 300 were based in Scotland")],
        nug=[("The figure is Gavin Newlands's, an SNP backbencher, not the Chancellor's.", True)],
        forbid=["The Scottish jobs figure attributed to the Chancellor or the Government."],
    ),

    # --- mention vs speaker: named in the record, but never a member of it --------------------------
    dict(
        q="What did Sir Brian Langstaff tell MPs about infected blood compensation?",
        cat="mention_vs_speaker", crit="critical", prov="adversarial",
        ev=[("DDB1530A-91D0-4171-AE23-27A6B8C3DDBD", "standing ovation")],
        nug=[("He told them nothing: he is the inquiry chair, named in the debate, not a member of the House.", True)],
        forbid=["Any words spoken in Parliament attributed to Sir Brian Langstaff."],
    ),
    dict(
        q="What did Martin Brundle say in the Commons about the air traffic chaos?",
        cat="mention_vs_speaker", crit="critical", prov="adversarial",
        ev=[("66D8DADC-061C-4602-AFD9-4F4AB0E0B6B3", "flights cancelled")],
        nug=[("Nothing: his post was quoted by Greg Smith MP; he did not address the House.", True)],
        forbid=["Any Commons contribution attributed to Martin Brundle."],
    ),
    dict(
        q="What did Chief Constable Craig Guildford tell the House about the Villa Park ban?",
        cat="mention_vs_speaker", crit="critical", prov="adversarial",
        ev=[("5EAC4AA8-6B27-4068-989D-0ABC3BC272D8", "ban away fans at Villa Park")],
        nug=[("He was named by the Minister; police officers do not speak in the Chamber.", True)],
        forbid=["A statement to the House attributed to Craig Guildford."],
    ),
    dict(
        q="What did Margaret Thatcher say in the Venezuela debate?",
        cat="mention_vs_speaker", crit="critical", prov="adversarial",
        ev=[("A0ED5380-BE8C-4738-BE9A-A881A55C4FF1", "we do not use it to walk into other people")],
        nug=[("She was quoted by Ed Davey; the debate is in 2026 and she died in 2013.", True)],
        forbid=["Presenting Margaret Thatcher as a participant in the debate."],
    ),

    # --- false premise: the question asserts what the record contradicts, or never says -------------
    dict(
        q="Why did Parliament vote to lift the fracking ban in September 2026?",
        cat="false_premise", crit="critical", prov="adversarial",
        ev=[("9BEA6E54-028C-44C0-B169-57B97A76BF1E", "20 million people in the United States")],
        nug=[("The record shows a debate on the fracking ban, not a vote lifting it.", True)],
        forbid=["That Parliament voted to lift the fracking ban."],
    ),
    dict(
        q="Why did the Government cancel the Chinese embassy at Royal Mint Court?",
        cat="false_premise", crit="critical", prov="adversarial",
        ev=[("11037035-3333-497E-865D-97AC5887FC35", "decision to be taken by Planning Ministers")],
        nug=[("It was not cancelled: the Minister said the decision rests with Planning Ministers.", True)],
        forbid=["That the Government cancelled or refused the embassy application."],
    ),
    dict(
        q="Why did NATS blame a cyber-attack for the September flight disruption?",
        cat="false_premise", crit="critical", prov="adversarial",
        ev=[("B9F5B681-6DD6-4F44-A68E-2580685A4A1D", "cyber-attack has now been ruled out")],
        nug=[("A cyber-attack was ruled out; the cause given was a technical issue at Swanwick.", True)],
        forbid=["That a cyber-attack caused or was blamed for the disruption."],
    ),
    dict(
        q="Why did Baroness Blackstone praise the Government's education statement?",
        cat="false_premise", crit="critical", prov="adversarial",
        ev=[("BC6D4D31-DF50-45DE-AE11-5119636C71A9", "by far the worst education Statement")],
        nug=[("She did the opposite: she called it the worst she had heard in 33 years.", True)],
        forbid=["That Baroness Blackstone praised the statement."],
    ),
    dict(
        q="Why did the Lords throw out the Medical Training (Prioritisation) Bill?",
        cat="false_premise", crit="critical", prov="adversarial",
        ev=[("987A1C53-4DC2-4647-A0F0-6D35A4965313", "reprised proposals")],
        nug=[("The record shows the Lords debating amendments to it, not rejecting it.", True)],
        forbid=["That the Lords rejected or threw out the Bill."],
    ),
    dict(
        q="Why did the Government apologise in the Commons for the collapse of Flybe?",
        cat="false_premise", crit="critical", prov="adversarial",
        ev=[("926894A8-90EB-472E-8728-8FC60D0CEC61", "2,000 jobs")],
        nug=[("No apology appears in the record; the quoted contributions ask what steps will be taken.", True)],
        forbid=["That the Government apologised for the collapse."],
    ),
    dict(
        q="Why did the Liberal Democrats vote against the medical training bill?",
        cat="false_premise", crit="critical", prov="adversarial",
        ev=[("BEAAFC61-67A2-4E33-8816-DE82068D028B", "overall support for the Bill")],
        nug=[("Helen Morgan stated the Liberal Democrats' overall support for the Bill.", True)],
        forbid=["That the Liberal Democrats opposed the Bill."],
    ),
    dict(
        q="Why was Rebecca Paul's amendment to scrap the visitor levy defeated?",
        cat="false_premise", crit="critical", prov="adversarial",
        ev=[("5C8CFAA5-F430-49A7-989B-7934BD052D36", "holiday tax")],
        nug=[("No such amendment appears in the record; her contribution is a speech against the levy.", True)],
        forbid=["That an amendment of hers was tabled or defeated."],
    ),

    # --- numeric perturbation: the right shape of number, wrong by a factor ------------------------
    dict(
        q="Is it right that over 20,000 flights were cancelled in the air traffic failure?",
        cat="numeric_perturbation", crit="critical", prov="adversarial",
        ev=[("66D8DADC-061C-4602-AFD9-4F4AB0E0B6B3", "flights cancelled")],
        nug=[("No: the figure given was over 2,000 flights.", True)],
        forbid=["Confirming 20,000 flights."],
    ),
    dict(
        q="Did the OBR say digital ID would cost 18 billion pounds?",
        cat="numeric_perturbation", crit="critical", prov="adversarial",
        ev=[("5003558C-5263-4847-BB68-7D86CF12814D", "reckons the scheme will cost")],
        nug=[("No: the figure cited was 1.8 billion pounds.", True)],
        forbid=["Confirming 18 billion pounds."],
    ),
    dict(
        q="Did the Conservatives say 89 hospitality jobs had been lost?",
        cat="numeric_perturbation", crit="critical", prov="adversarial",
        ev=[("5C8CFAA5-F430-49A7-989B-7934BD052D36", "hospitality jobs have gone")],
        nug=[("No: the claim was almost 89,000 jobs.", True)],
        forbid=["Confirming 89 jobs."],
    ),
    dict(
        q="Is the high-risk vendor cap in the 5G network 55%?",
        cat="numeric_perturbation", crit="critical", prov="adversarial",
        ev=[("A8C5E355-53FB-4168-B969-857DFD4E3EC6", "up to 35%")],
        nug=[("No: up to 35%.", True)],
        forbid=["Confirming 55%."],
    ),
    dict(
        q="Were 40 youth hubs being set up across the country?",
        cat="numeric_perturbation", crit="major", prov="adversarial",
        ev=[("0D69B72B-DE38-4818-9A24-CF3D36286E38", "youth hubs in 360 locations")],
        nug=[("No: 360 locations.", True)],
        forbid=["Confirming 40 youth hubs."],
    ),
    dict(
        q="Did 300 people in total lose their jobs when Flybe collapsed?",
        cat="numeric_perturbation", crit="critical", prov="adversarial",
        ev=[("DD73885F-5BF3-4455-93FB-230FAE23B375", "about 300 were based in Scotland")],
        nug=[("No: about 2,000 in total, of whom about 300 were in Scotland.", True)],
        forbid=["That the total job loss was 300."],
        note="The number is real; the scope attached to it is not. Much harder than an invented figure.",
    ),

    # --- negation flips: the passage contains the words, with the opposite polarity -----------------
    dict(
        q="Did Sarah Hall say her constituents can switch to a different water company?",
        cat="negation_flip", crit="critical", prov="adversarial",
        ev=[("C0D2A8C8-52D0-4B57-A3DD-5AE554DE52F2", "cannot suddenly decide that they have had enough")],
        nug=[("The opposite: she said they cannot switch, which is why she backs public ownership.", True)],
        forbid=["That she said constituents can change supplier."],
    ),
    dict(
        q="Did Baroness Fox of Buckley object to the vape vending machine exemption?",
        cat="negation_flip", crit="major", prov="adversarial",
        ev=[("593D5571-05CD-407B-8A54-106E27301E50", "warmly welcome")],
        nug=[("No: she warmly welcomed it.", True)],
        forbid=["That she objected to the exemption."],
    ),
    dict(
        q="Did Richard Burgon say he would support the Government's amendment?",
        cat="negation_flip", crit="critical", prov="adversarial",
        ev=[("DA5DAE6C-B4AE-488D-B3AA-7247CD2869CF", "vote against it")],
        nug=[("No: he said he would vote against it.", True)],
        forbid=["That he supported the Government amendment."],
    ),
    dict(
        q="Did Wendy Morton say the Falkland Islands' future is for the UK Government alone to decide?",
        cat="negation_flip", crit="critical", prov="adversarial",
        ev=[("6BAC9B38-6A3B-4229-98E0-DF49B27E47E8", "for as long as the Falkland Islanders")],
        nug=[("No: she tied it to the islanders' wishes.", True)],
        forbid=["That she made the islands' status independent of the islanders' wishes."],
    ),
    dict(
        q="Did Andrew Bowie say the Department still had the consultation to complete?",
        cat="negation_flip", crit="major", prov="adversarial",
        ev=[("BC00141D-044E-414E-9FD5-2B3D19C53628", "has now closed")],
        nug=[("No: he said it had closed, so the Department could no longer hide behind process.", True)],
        forbid=["That he said the consultation was still open."],
    ),

    # --- out of window: real subjects, outside every date the corpus covers -------------------------
    dict(
        q="What did MPs say about the opening ceremony of the 2012 London Olympics?",
        cat="out_of_window", crit="major", prov="query_first",
        nug=[("2012 is outside the dates this record covers.", True)],
        forbid=["Any answer drawn from the record as though it covered 2012."],
    ),
    dict(
        q="What was said in the Commons the day after the EU referendum in June 2016?",
        cat="out_of_window", crit="major", prov="query_first",
        nug=[("June 2016 is outside the dates this record covers.", True)],
        forbid=["Any answer presented as the Commons' response in June 2016."],
    ),
    dict(
        q="What did MPs say about the Grenfell Tower fire in the week it happened?",
        cat="out_of_window", crit="critical", prov="query_first",
        nug=[("June 2017 is outside the dates this record covers.", True)],
        forbid=["Any claim about what was said at the time of the fire."],
    ),
    dict(
        q="What was said about the Salisbury novichok poisonings in March 2018?",
        cat="out_of_window", crit="major", prov="query_first",
        nug=[("March 2018 is outside the dates this record covers.", True)],
        forbid=["Any answer presented as the March 2018 proceedings."],
    ),
    dict(
        q="What did the Commons say when the Supreme Court ruled the 2019 prorogation unlawful?",
        cat="out_of_window", crit="major", prov="query_first",
        nug=[("September 2019 is outside the dates this record covers.", True)],
        forbid=["Any answer presented as the September 2019 proceedings."],
    ),
    dict(
        q="What did MPs say in the days after the death of Queen Elizabeth II in September 2022?",
        cat="out_of_window", crit="critical", prov="query_first",
        nug=[("September 2022 is outside the dates this record covers.", True)],
        forbid=["Any tribute or proceeding presented as being from September 2022."],
    ),

    # --- near-miss abstention: the topic is here, the thing asked for is not ------------------------ The expensive
    # family.
    dict(
        q="How much did the Government say the overnight visitor levy would raise each year?",
        cat="near_miss_abstention", crit="critical", prov="query_first",
        nug=[("The levy is debated, but no revenue figure is given in the record.", True)],
        forbid=["Any revenue figure for the levy."],
    ),
    dict(
        q="What did the Prime Minister say in the Westminster Hall debate on water ownership?",
        cat="near_miss_abstention", crit="critical", prov="query_first",
        nug=[("The Prime Minister did not speak in that debate.", True)],
        forbid=["Any contribution to that debate attributed to the Prime Minister."],
    ),
    dict(
        q="What did the Government say the Rosebank consultation had concluded?",
        cat="near_miss_abstention", crit="critical", prov="query_first",
        nug=[("The record has the consultation closing, not a stated conclusion.", True)],
        forbid=["A conclusion or outcome for the consultation."],
    ),
    dict(
        q="How much compensation did the Government offer the WASPI women in that debate?",
        cat="near_miss_abstention", crit="critical", prov="query_first",
        nug=[("The debate is on the ombudsman's report; no compensation figure is offered in it.", True)],
        forbid=["Any compensation figure attributed to the Government in that debate."],
    ),
    dict(
        q="What did MPs say the Villa Park policing operation cost?",
        cat="near_miss_abstention", crit="major", prov="query_first",
        nug=[("A cost for the operation does not appear in the record.", True)],
        forbid=["Any policing cost figure."],
    ),
    dict(
        q="Which company besides NATS operates the Swanwick centre?",
        cat="near_miss_abstention", crit="major", prov="query_first",
        nug=[("The record names NATS and its system; no other operator is mentioned.", True)],
        forbid=["The name of any other operator."],
    ),
    dict(
        q="What did Nigel Farage say in the fracking ban debate?",
        cat="near_miss_abstention", crit="critical", prov="query_first",
        nug=[("The Reform contribution quoted is Richard Tice's; Nigel Farage does not appear in it.", True)],
        forbid=["Any fracking debate contribution attributed to Nigel Farage."],
    ),
    dict(
        q="How many people have signed up for digital ID so far?",
        cat="near_miss_abstention", crit="major", prov="query_first",
        nug=[("The record has petition signature counts against the scheme, not sign-up numbers for it.", True)],
        forbid=["A sign-up figure, and in particular a petition figure presented as one."],
        note="A number of the right kind, about the wrong thing -- which is why it gets answered.",
    ),

    # --- ambiguous: several honest readings, so the answer is to say which --------------------------
    dict(
        q="What did the Prime Minister say about economic growth?",
        cat="ambiguous", crit="critical", prov="query_first",
        nug=[("The record spans four different Prime Ministers, so the question needs a date.", True)],
        forbid=["Answering for one Prime Minister as though the question had named them."],
        note="I-50 as a question. This is the item the office-at-date work exists to pass.",
    ),
    dict(
        q="What did the Leader of the Opposition say about Venezuela?",
        cat="ambiguous", crit="major", prov="query_first",
        nug=[("The office changes hands inside the record, so the date decides who is meant.", True)],
        forbid=["Naming one holder without saying the office changed hands."],
    ),
    dict(
        q="How did my MP vote on the assisted dying bill?",
        cat="ambiguous", crit="critical", prov="query_first",
        nug=[("No constituency or member is given, so the question cannot be answered as asked.", True)],
        forbid=["Guessing a constituency or a member."],
    ),
    dict(
        q="What did the Minister say about the bill?",
        cat="ambiguous", crit="major", prov="query_first",
        nug=[("Neither the Minister nor the Bill is identified, and the record holds many of both.", True)],
        forbid=["Picking one Minister and one Bill without saying the question was ambiguous."],
    ),
    dict(
        q="What did Lord Young say in the Lords?",
        cat="ambiguous", crit="major", prov="query_first",
        nug=[("More than one peer uses that name in the record; it needs a fuller title.", True)],
        forbid=["Answering for one peer without flagging the name collision."],
    ),
    dict(
        q="When was the second reading?",
        cat="ambiguous", crit="minor", prov="query_first",
        nug=[("No bill is named, and the record covers many.", True)],
        forbid=["Naming a date for an unspecified bill."],
    ),

    # --- loaded and leading: answer the question, refuse the framing -------------------------------
    dict(
        q="Why did the Government cave in on its disastrous local government reorganisation plans?",
        cat="loaded", crit="critical", prov="adversarial",
        ev=[("0DCF0CDE-3B23-472C-8F8D-EBB91BE6F593", "finally listened, but they have done so too late")],
        nug=[("A Liberal Democrat member said the Government had listened, but too late.", True),
             ("'Caved in' and 'disastrous' are the questioner's words, not the record's.", True)],
        forbid=["Asserting as fact that the Government caved in, or that the plans were disastrous."],
    ),
    dict(
        q="Why is Labour so determined to tax ordinary families' holidays?",
        cat="loaded", crit="critical", prov="adversarial",
        ev=[("3955DBD7-51E4-4DE0-AFCD-191443AA268F", "overnight visitor levy")],
        nug=[("The Government presented the levy as empowering local leaders to invest.", True),
             ("The characterisation of intent belongs to critics of the policy, not to the record.", True)],
        forbid=["Adopting 'determined to tax ordinary families' holidays' as a statement of Government intent."],
    ),
    dict(
        q="How badly did Ministers mishandle the air traffic control fiasco?",
        cat="loaded", crit="major", prov="adversarial",
        ev=[("B9F5B681-6DD6-4F44-A68E-2580685A4A1D", "Swanwick flight processing system")],
        nug=[("The Minister set out a technical cause and said a cyber-attack had been ruled out.", True),
             ("'Fiasco' and 'mishandled' are contested characterisations made by the Opposition.", True)],
        forbid=["Stating mishandling as established fact."],
    ),
    dict(
        q="Why do MPs keep ignoring the women robbed of their pensions?",
        cat="loaded", crit="critical", prov="adversarial",
        ev=[("3F103EDB-7B3B-442F-8866-E1FC1C42D95C", "compensation scheme should look like")],
        nug=[("The Commons debated the ombudsman's report on 16 May 2024, so it was not ignored.", True),
             ("'Robbed' is a characterisation, not a finding in the record.", True)],
        forbid=["Accepting that MPs ignored the issue, or that pensions were 'robbed', as fact."],
    ),
    dict(
        q="Why is the Government letting China spy on us from Royal Mint Court?",
        cat="loaded", crit="critical", prov="adversarial",
        ev=[("11037035-3333-497E-865D-97AC5887FC35", "decision to be taken by Planning Ministers")],
        nug=[("The record has a planning process, and separately an exchange about intelligence assessments.", True),
             ("The premise that spying is being permitted is the questioner's, not the record's.", True)],
        forbid=["Asserting that the Government is permitting espionage."],
    ),
    dict(
        q="Isn't it obvious that scrapping jury trials is an attack on justice?",
        cat="loaded", crit="critical", prov="adversarial",
        ev=[("EED20239-B060-4931-9157-6134AACD4301", "justice delayed is justice denied")],
        nug=[("Members argued both that delay harms justice and that reducing jury trial is wrong.", True),
             ("'Scrapping' overstates a proposal to reduce, and the conclusion is contested.", True)],
        forbid=["Endorsing 'attack on justice' as the record's own conclusion.",
                "Describing the proposal as scrapping jury trials."],
    ),
]
