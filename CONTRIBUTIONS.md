# Contributions

Bradley Yong is the project owner and proposal/presentation author. Source,
experiment automation, analysis and documentation were developed with AI coding
assistance under the owner's direction. Human review and explanation remain
the owner's responsibility.

This repository preparation imports the verified course-project export at
`ca985876a6539cf4630091d72554a91501ae3eef`. That separate export contains six real
delivery commits. Those commits are not ancestry of this repository's initial
commit and are not evidence of a hand-written development timeline.

The import preserves runtime, test and cached-reproduction files. One trainer
now validates embedded dataset metadata before importing optional PyTorch;
its existing rejection tests therefore also work in a lightweight CPU install.
The training settings and fitting logic are unchanged.
Public-facing documentation is revised, historical export receipts move to
`docs/provenance/`, and the verified report/demo are included. No new training,
inference or physical race is implied by this packaging work.

No earlier experiment commits, team members, dates or contribution percentages
are fabricated. The current Git history records only work actually committed
in this repository. See [source provenance](docs/provenance/README.md).
