# Heading labels (to fill in)

`heading_labels.csv` lists the 70 most common headings in each dataset (GovReport, QASPER,
Natural Questions), with how many documents contain each one and an example of its text.

Fill the `label` column with one of:

- `functional`: the heading tells you what *kind* of content the section holds, in a way a
  question could target, even though many documents share it. Examples: "Results", "Datasets",
  "Experimental Setup", "Early life", "Plot", "Recommendations for Executive Action".
- `boilerplate`: a standard administrative or navigational part that questions rarely target, or
  a placeholder. Examples: "GAO Contact", "Staff Acknowledgments", "References", "See also",
  "External links", "Untitled 1".
- `unsure`: you cannot decide (add a note).

Test to apply: *if you only saw this heading, would it help you guess whether a typical question
about this kind of document is answered in this section?* Yes → functional, no → boilerplate.

Use the example text when the heading alone is ambiguous. Label each row independently. It takes
about an hour. If possible, ask a second person to label the same file separately (save as
`heading_labels_rater2.csv`) so agreement can be reported.

How it is used: the specificity fallback (which makes common headings neutral) will be applied
only to headings labelled boilerplate, to test whether that keeps the gains on functional headings
such as QASPER's "Results" while removing the losses on boilerplate ones.
