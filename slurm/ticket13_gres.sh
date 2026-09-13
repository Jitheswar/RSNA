#!/usr/bin/env bash

# Slurm may expose the controller's gres.conf with an explicit NodeName field,
# while slurmd's local copy deliberately contains only the entries for the
# active node.  Keep the distinction explicit so the caller can record which
# form supplied the device mapping.

_ticket13_node_name_matches() {
  local pattern="$1"
  local actual="$2"

  [[ -n "${pattern}" && -n "${actual}" ]] || return 1
  [[ "${pattern}" == "${actual}" ]] && return 0

  if [[ "${pattern}" =~ ^([^[]*)\[([^]]+)\]([^[]*)$ ]]; then
    local prefix="${BASH_REMATCH[1]}"
    local members="${BASH_REMATCH[2]}"
    local suffix="${BASH_REMATCH[3]}"
    [[ "${actual}" == "${prefix}"*"${suffix}" ]] || return 1

    local middle="${actual#"${prefix}"}"
    middle="${middle%"${suffix}"}"
    local member
    local first_text
    local last_text
    local first
    local last
    local candidate
    local -a member_values
    IFS=',' read -r -a member_values <<< "${members}"
    for member in "${member_values[@]}"; do
      if [[ "${member}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        first_text="${BASH_REMATCH[1]}"
        last_text="${BASH_REMATCH[2]}"
        [[ "${middle}" =~ ^[0-9]+$ ]] || continue
        first=$((10#${first_text}))
        last=$((10#${last_text}))
        candidate=$((10#${middle}))
        if (( first <= candidate && candidate <= last )); then
          return 0
        fi
      elif [[ "${middle}" == "${member}" ]]; then
        return 0
      fi
    done
  fi
  return 1
}

ticket13_collect_gres_config() {
  local node_name="${1:-}"
  local config_path="${2:-}"
  local gres_line
  local configured_node
  local device_file
  local config_scope
  local match_count=0

  TICKET13_GRES_CONFIG_ENTRY=""
  TICKET13_GRES_CONFIG_SCOPE=""
  TICKET13_GRES_DEVICE_FILE=""

  [[ -n "${node_name}" && -r "${config_path}" ]] || return 0
  while IFS= read -r gres_line; do
    [[ "${gres_line}" =~ ^[[:space:]]*(#|$) ]] && continue
    [[ "${gres_line}" =~ (^|[[:space:]])Name=gpu([[:space:]]|$) ]] || continue
    [[ "${gres_line}" =~ (^|[[:space:]])File=([^[:space:]]+) ]] || continue
    device_file="${BASH_REMATCH[2]}"

    if [[ "${gres_line}" =~ (^|[[:space:]])NodeName=([^[:space:]]+)([[:space:]]|$) ]]; then
      configured_node="${BASH_REMATCH[2]}"
      _ticket13_node_name_matches "${configured_node}" "${node_name}" || continue
      config_scope="named_node"
    else
      # This file was read from the active compute node, so a NodeName-less
      # entry is scoped to that node by slurmd's local configuration.
      config_scope="local_node"
    fi

    match_count=$((match_count + 1))
    if (( match_count > 1 )); then
      TICKET13_GRES_CONFIG_ENTRY=""
      TICKET13_GRES_CONFIG_SCOPE=""
      TICKET13_GRES_DEVICE_FILE=""
      return 1
    fi
    TICKET13_GRES_CONFIG_ENTRY="${gres_line}"
    TICKET13_GRES_CONFIG_SCOPE="${config_scope}"
    TICKET13_GRES_DEVICE_FILE="${device_file}"
  done < "${config_path}"
}
