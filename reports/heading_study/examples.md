# Heading study: examples

Within-document retrieval. SA = structure-aware (alpha = 0.2); flat = BM25 + dense over text only. w = heading specificity (0 = heading shared by every document, 1 = unique). Questions are shown as written; ICT questions are sentences removed from the target section.


## GovReport (ICT)


### Generic target heading, structure-aware loses

- **Q:** To inform our understanding of the prohibition on NNSA’s expenditures on nuclear security in Russia, we reviewed laws since fiscal year 2015 that restricted relevant NNSA funding in some way.
  - Target: Appendix I: Objectives, Scope, and Methodology (w = 0.33)
  - Flat top-1: According to Stakeholders, Opportunities May Exist for Cooperation to Improve Russian Nuclear Material Security, but Such Cooperation Would Face Challenges > Potential Cooperation Faces Significant Challenges (w = 1.00)
  - Structure-aware top-1: According to Stakeholders, Opportunities May Exist for Cooperation to Improve Russian Nuclear Material Security, but Such Cooperation Would Face Challenges > Potential Cooperation Faces Significant Challenges (w = 1.00)
- **Q:** Beneficiaries with mild behavioral health conditions—such as mild depression—may require less complex care than beneficiaries with serious behavioral health conditions—such as schizophrenia—or with multiple interacting behavioral or physical health conditions.
  - Target: Background (w = 0.11)
  - Flat top-1: Background (w = 0.11)
  - Structure-aware top-1: Background > Provision of Information to Medicare Beneficiaries (w = 1.00)
- **Q:** With regard to the recommendation on developing controls for virtual workspace access (recommendation 11), DHS stated that CBP believes that furloughed employees must be able to passively monitor the status of the government shutdown and access important agency communications using DHS-issued electronic devices.
  - Target: Agency Comments and Our Evaluation (w = 0.27)
  - Flat top-1: Agency Comments and Our Evaluation (w = 0.27)
  - Structure-aware top-1: Two of Four Selected Agency Components Documented Shutdown Procedures, and None Had Sufficient Controls for Workspace Access during a Shutdown > Selected Agency Components Generally Tracked Employees Working but Did Not Have Controls for Workspace Access during the Shutdown > Selected Agency Components Had Insufficient Controls for Physical and Virtual Workspace Access (w = 1.00)

### Specific target heading, structure-aware wins

- **Q:** As of March 2019, FDA data showed that India and China had the most manufacturing establishments shipping drugs to the United States, with about 40 percent of all foreign establishments in these two countries.
  - Target: Background Globalization of Drug Manufacturing (w = 1.00)
  - Flat top-1: The Number of Foreign Inspections Declined in Recent Years, and the Majority of Such Inspections Identified Deficiencies Total Number of FDA Foreign Drug Inspections Has Decreased Since Fiscal Year 2016 after Several Years of Increases (w = 1.00)
  - Structure-aware top-1: The Number of Foreign Inspections Declined in Recent Years, and the Majority of Such Inspections Identified Deficiencies Total Number of FDA Foreign Drug Inspections Has Decreased Since Fiscal Year 2016 after Several Years of Increases (w = 1.00)
- **Q:** 87-195; as amended) first authorized research on family planning issues, among many other things, and in 1965 Congress authorized USAID to create contraceptive distribution programs through the Office of Population.
  - Target: Untitled 1 > Introduction > Evolution of U.S. Policy and Programs (w = 1.00)
  - Flat top-1: Untitled 1 > Introduction (w = 0.26)
  - Structure-aware top-1: Untitled 1 > Introduction > Evolution of U.S. Policy and Programs (w = 1.00)
- **Q:** On the other hand, if a program does not meet all of the criteria then it “partially met” the criteria for the accreditation standard.
  - Target: The Military Services’ PME Programs Are Accredited, but Not All Programs Met the JPME Seminar Student Mix Accreditation Requirement The Military Services’ Intermediate- and Senior- level PME Programs Are Accredited to Award Master’s Degrees > The Military Services’ Intermediate- and Senior- level PME Programs Are Accredited to Award JPME Credit, but Not All Programs Met the Seminar Student Mix Requirement All Military Service Intermediate- and Senior-level PME Programs Are Accredited to Award JPME Credit (w = 1.00)
  - Flat top-1: The Military Services’ PME Programs Are Accredited, but Not All Programs Met the JPME Seminar Student Mix Accreditation Requirement The Military Services’ Intermediate- and Senior- level PME Programs Are Accredited to Award Master’s Degrees > The Military Services’ Intermediate- and Senior- level PME Programs Are Accredited to Award JPME Credit, but Not All Programs Met the Seminar Student Mix Requirement All Military Service Intermediate- and Senior-level PME Programs Are Accredited to Award JPME Credit > Most Military Services’ Senior- level PME Programs Met the JPME Seminar Student Mix Accreditation Requirement, but Some Intermediate-level Programs Did Not (w = 1.00)
  - Structure-aware top-1: The Military Services’ PME Programs Are Accredited, but Not All Programs Met the JPME Seminar Student Mix Accreditation Requirement The Military Services’ Intermediate- and Senior- level PME Programs Are Accredited to Award Master’s Degrees (w = 1.00)

## GovReport (LLM-written)


### Generic target heading, structure-aware loses

- **Q:** Why might beneficiaries with substance use disorders not seek available Medicare coverage for their treatment?
  - Target: Conclusions (w = 0.18)
  - Flat top-1: CMS Uses Various Approaches to Provide Coverage Information to Beneficiaries, but Annual Mailing Does Not Include Explicit Information on SUD Treatment Coverage > CMS’s Annual Mailing to Beneficiaries Does Not Include Explicit Information on Medicare Coverage for SUD Treatment (w = 1.00)
  - Structure-aware top-1: CMS Uses Various Approaches to Provide Coverage Information to Beneficiaries, but Annual Mailing Does Not Include Explicit Information on SUD Treatment Coverage > CMS’s Annual Mailing to Beneficiaries Does Not Include Explicit Information on Medicare Coverage for SUD Treatment (w = 1.00)
- **Q:** How does the VA plan to address the issues identified in recommendations 3 and 4?
  - Target: Agency Comments and Our Evaluation (w = 0.27)
  - Flat top-1: Agency Comments and Our Evaluation (w = 0.27)
  - Structure-aware top-1: Recommendations for Executive Action (w = 0.24)
- **Q:** How can TSA improve coordination with air carriers and stakeholders when issuing security directives for international flights?
  - Target: Conclusions (w = 0.18)
  - Flat top-1: TSA Reviews Directives, but Industry Coordination Is Inconsistent, and TSA Has Not Fully Developed Procedures for Incorporating Directives into Security Programs > TSA Does Not Consistently Coordinate with Air Carriers and Other Industry Stakeholders When Reviewing Directives (w = 1.00)
  - Structure-aware top-1: TSA Reviews Directives, but Industry Coordination Is Inconsistent, and TSA Has Not Fully Developed Procedures for Incorporating Directives into Security Programs > TSA Does Not Consistently Coordinate with Air Carriers and Other Industry Stakeholders When Reviewing Directives (w = 1.00)

### Specific target heading, structure-aware wins

- **Q:** How did the AOA study team gather and integrate input from both military and commercial stakeholders to shape its alternatives?
  - Target: In Accordance with Its Study Plan, DOD Used Multiple Methods to Obtain Stakeholder Input > Military and Commercial Stakeholders Provided Input to the AOA (w = 1.00)
  - Flat top-1: Background > AOA Process and Best Practices (w = 1.00)
  - Structure-aware top-1: In Accordance with Its Study Plan, DOD Used Multiple Methods to Obtain Stakeholder Input > Military and Commercial Stakeholders Provided Input to the AOA (w = 1.00)
- **Q:** How did the Bureau address staffing shortages during the census address canvassing operation?
  - Target: The Bureau Had Successes and Challenges during In- Field Canvassing, Which Have Potential Implications for Future Operations > The Bureau Is Taking Steps to Address Challenges with Hiring and Onboarding Staff (w = 1.00)
  - Flat top-1: The Bureau Completed In-Field Address Canvassing on Schedule and under Budget, but Listers Did Not Always Follow Procedures > Productivity Was Higher Than Expected (w = 1.00)
  - Structure-aware top-1: The Bureau Had Successes and Challenges during In- Field Canvassing, Which Have Potential Implications for Future Operations > The Bureau Is Taking Steps to Address Challenges with Hiring and Onboarding Staff (w = 1.00)
- **Q:** Should Congress reassess the ongoing justification, funding, and risks of a covert operation given potential changes in its environment and effectiveness?
  - Target: Untitled 1 > An Iterative Process > Questions for Congress (w = 1.00)
  - Flat top-1: Untitled 1 > A Framework for Oversight; Lines of Inquiry for Congress (w = 1.00)
  - Structure-aware top-1: Untitled 1 > A Framework for Oversight; Lines of Inquiry for Congress (w = 1.00)

## QASPER


### Generic target heading, structure-aware loses

- **Q:** How does their BERT-based model work?
  - Target: Introduction (w = 0.01)
  - Flat top-1: Query-Focused Summarization Model (w = 1.00)
  - Structure-aware top-1: Query-Focused Summarization Model (w = 1.00)
- **Q:** What metrics do they use?
  - Target: Introduction (w = 0.01)
  - Flat top-1: Validation for Selection of Hyper-parameters (w = 1.00)
  - Structure-aware top-1: Results (w = 0.22)
- **Q:** What is the accuracy of this model compared to sota?
  - Target: Introduction (w = 0.01)
  - Flat top-1: Conclusion (w = 0.11)
  - Structure-aware top-1: Mapping coherence (w = 1.00)

### Specific target heading, structure-aware wins

- **Q:** How many steps of BPE do they experiment with?
  - Target: Choosing number of BPE merges (w = 1.00)
  - Flat top-1: Joint bilingual learning of BPE units (w = 1.00)
  - Structure-aware top-1: BPE for related languages (w = 1.00)
- **Q:** What is the size of Multi-news dataset?
  - Target: Statistics and Analysis (w = 1.00)
  - Flat top-1: Conclusion (w = 0.11)
  - Structure-aware top-1: Multi-News Dataset (w = 1.00)
- **Q:** What other model inference optimization schemes authors explore?
  - Target: Related Work > Residual Learning (w = 1.00)
  - Flat top-1: Inference Performance Optimization (w = 1.00)
  - Structure-aware top-1: Inference Performance Optimization (w = 1.00)

## Natural Questions


### Generic target heading, structure-aware loses

- **Q:** how many pennsylvania house of representatives are there
  - Target: Introduction (w = 0.00)
  - Flat top-1: Introduction (w = 0.00)
  - Structure-aware top-1: Members of the Pennsylvania House of Representatives (w = 1.00)
- **Q:** when did korea separate into north and south
  - Target: Introduction (w = 0.00)
  - Flat top-1: Introduction (w = 0.00)
  - Structure-aware top-1: Korean War (w = 1.00)
- **Q:** what is alpha centauri's approximate distance from earth
  - Target: Introduction (w = 0.00)
  - Flat top-1: Observational History (w = 1.00)
  - Structure-aware top-1: Binary system > Proxima Centauri (w = 1.00)

### Specific target heading, structure-aware wins

- **Q:** when did university stop being free in australia
  - Target: History > Introduction of HECS (w = 1.00)
  - Flat top-1: History > Abolition of university fees (w = 1.00)
  - Structure-aware top-1: History > Abolition of university fees (w = 1.00)
- **Q:** what is the purpose of the muscles that line the esophagus
  - Target: Function > Swallowing (w = 1.00)
  - Flat top-1: Structure > Nerve supply (w = 0.91)
  - Structure-aware top-1: Structure > Nerve supply (w = 0.91)
- **Q:** what teams play in spring training in arizona
  - Target: Spring training locations by team > Cactus League (w = 1.00)
  - Flat top-1: Spring training locations by team (w = 1.00)
  - Structure-aware top-1: Spring training locations by team (w = 1.00)
