#!/usr/bin/env bash
# Run only in a disposable checkout, with origin pointing at GITHUB_REPOSITORY.
set -euo pipefail

: "${GITHUB_REPOSITORY:?Set the destination owner/repository}"
: "${UPSTREAM_REPOSITORY:?Set the upstream owner/repository}"
: "${SYNC_AUTHOR:?Set the PR author login}"

git config user.name 'Railgun-wiki'
git config user.email '64968531+Railgun-wiki@users.noreply.github.com'
git config core.hooksPath /dev/null
git config commit.gpgsign false
git config merge.gpgsign false

summary_file="${GITHUB_STEP_SUMMARY:-/dev/stdout}"
failures=0
printf '## Upstream synchronization\n\n' >> "$summary_file"

# Query before changing any refs; API errors must not look like zero open PRs.
# Filter by both author and head repository, including future PR branches.
pulls=$(gh api --paginate --method GET \
  "repos/$UPSTREAM_REPOSITORY/pulls" -f state=open -f per_page=100 \
  --jq ".[] | select(.user.login == \"$SYNC_AUTHOR\" and .head.repo.full_name == \"$GITHUB_REPOSITORY\") | [.head.ref, .base.ref] | @tsv")

sync_branch() {
  local branch=$1 base=$2 source destination before after
  # Branch names come from GitHub, but still validate before using as refspecs.
  git check-ref-format "refs/heads/$branch" || return 1
  git check-ref-format "refs/heads/$base" || return 1
  source="refs/sync-upstream/source"
  destination="refs/sync-upstream/destination"
  if ! git fetch --no-tags "https://github.com/$UPSTREAM_REPOSITORY.git" \
    "+refs/heads/$base:$source"; then
    printf -- '- `%s`: failed to fetch upstream `%s`.\n' "$branch" "$base" >> "$summary_file"
    return 1
  fi
  if ! git fetch --no-tags origin "+refs/heads/$branch:$destination"; then
    printf -- '- `%s`: failed to fetch destination.\n' "$branch" >> "$summary_file"
    return 1
  fi
  before=$(git rev-parse "$destination")
  git checkout --detach "$before" || return 1
  # No rebase or force push: preserve fork-only commits (including this workflow).
  if ! git -c merge.autoStash=false merge --no-edit "$source"; then
    git merge --abort || true
    printf -- '- `%s`: merge failed against upstream `%s`; remote unchanged.\n' "$branch" "$base" >> "$summary_file"
    return 1
  fi
  after=$(git rev-parse HEAD)
  if [[ "$before" == "$after" ]]; then
    printf -- '- `%s`: already up to date with `%s`.\n' "$branch" "$base" >> "$summary_file"
    return 0
  fi
  # Ordinary push rejects concurrent non-fast-forward changes automatically.
  if ! git push origin "$after:refs/heads/$branch"; then
    printf -- '- `%s`: push rejected; inspect permissions or concurrent updates.\n' "$branch" >> "$summary_file"
    return 1
  fi
  printf -- '- `%s`: merged upstream `%s` and pushed.\n' "$branch" "$base" >> "$summary_file"
}

sync_branch main main || failures=$((failures + 1))
while IFS=$'\t' read -r branch base; do
  [[ -n "$branch" ]] || continue
  # main was handled above; do not merge a different PR base into it.
  [[ "$branch" != main ]] || continue
  sync_branch "$branch" "$base" || failures=$((failures + 1))
done < <(printf '%s\n' "$pulls" | sort -u)

if (( failures > 0 )); then
  printf '\nFailed branches: %s. Resolve conflicts/permissions and rerun.\n' "$failures" >> "$summary_file"
  exit 1
fi
