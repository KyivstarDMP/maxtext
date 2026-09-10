# MaxText: Upstream Contribution Runbook

Develop against `develop`. Prepare upstream contributions by copying selected changes onto `upstream/main`; internal delivery does not depend on upstream acceptance.

- `origin` is our fork; `upstream` is the original MaxText repository.
- `develop` contains our custom version and receives upstream updates through merges.
- Branch names below are examples. Internal naming and feature integration policies are outside this runbook.

Start commands with a clean working tree. Replace uppercase SHA placeholders with actual commit hashes.

## 1. Select Changes

The source feature may be unmerged or already integrated into `develop`.

**Prefer retaining the source branch at its final feature tip after merging into `develop`, until the upstream contribution is complete.** This preserves the commits for selection; rebasing onto `develop` afterward can remove the feature-only commit range.

Select explicit SHAs from one or more source branches or internal PRs, in dependency order:

- Include the feature and required prerequisites; adapt dependencies on custom code for upstream.
- Exclude unrelated changes, changes already upstream, and merge commits. Check whether merge resolutions contain adjustments that must be carried over separately.
- Do not assume the latest `n` commits are the feature. `develop..feature/abc` may be empty after integration.

## 2. Open the Upstream PR

A coding agent can prepare the upstream PR, including identifying the relevant commits (`SOURCE_SHA_1`, `SOURCE_SHA_2`, etc.) and cherry-picking them in dependency order.

```bash
git fetch upstream
git switch -c pr/abc upstream/main
git cherry-pick SOURCE_SHA_1 SOURCE_SHA_2
```

Review `git diff upstream/main...HEAD` and run the relevant upstream tests. Confirm that the contribution works without custom code and excludes internal-only files, including this runbook.

```bash
git push -u origin pr/abc
```

Open a PR to the original repository's `main`, following its contribution requirements.

Never merge `develop` or a source feature branch into the upstream PR branch. Keep its history linear.

If cherry-picking conflicts, resolve the files, stage them with `git add`, and run `git cherry-pick --continue`. Use `git cherry-pick --abort` to cancel.

## 3. Update During Review

Make review changes on `pr/abc`. Commit review bug fixes separately so they can be cherry-picked internally. Squash them into the upstream feature only after any required internal copies have been made.

When an upstream update is needed:

```bash
git fetch upstream
git switch pr/abc
git rebase upstream/main
```

Resolve and stage conflicts, then use `git rebase --continue`; use `git rebase --abort` to cancel. Retest before pushing. Push new commits with `git push origin pr/abc`; after rebasing, amending, or squashing published commits, use `git push --force-with-lease origin pr/abc`.

## 4. Bring Review Fixes Back Internally

Copy required fixes using their current SHAs on the PR branch.

If the source feature is still unmerged, apply the fix there:

```bash
git switch feature/abc
git cherry-pick REVIEW_FIX_SHA
```

If the feature is already merged, start new work from current `origin/develop`:

```bash
git fetch origin
git switch -c fix/abc origin/develop
git cherry-pick REVIEW_FIX_SHA
```

Adapt the fix and include prerequisites if needed. Test against our custom version and integrate through the normal internal process.

If the fix was folded into an amended or squashed feature commit, extract only the fix as a new internal commit. Do not replay the entire feature. Likewise, explicitly copy any new internal fixes needed by the upstream PR.

Do not merge `pr/abc` directly into `develop`. Transfer individual fixes; accepted upstream changes arrive through the upstream synchronization process below.

## 5. Synchronize `develop` With Upstream

Synchronize periodically, including after upstream PR acceptance. This is independent of feature development and PR preparation.

```bash
git fetch --multiple origin upstream
git switch develop
git merge --ff-only origin/develop
git merge upstream/main
```

Resolve conflicts and validate the updated custom version with automated checks and a representative training smoke test where practical. Publish through the normal internal process only after validation. Preserve upstream ancestry when integrating the result: do not squash the upstream synchronization or rebase `develop`.

Keep earlier copied commits in history; upstream acceptance does not require reverting them. Identical edits normally merge without duplication. Validate review changes and custom adaptations even if the merge has no conflicts.

Deploy tested `develop` revisions and record the full commit SHA from `git rev-parse HEAD` in the deployed checkout.
