#!/bin/bash

# ==============================================================================
# GUI-Agent All-Domains Runner Script
# ==============================================================================
# This script runs evaluations for ALL domains under a given evaluation type
# and saves a combined success rate summary.
#
# Usage:
#   ./run_all_domains.sh [options]
#
# Examples:
#   ./run_all_domains.sh                                  # Run webvoyager (default) with all domains
#   ./run_all_domains.sh --model qwen2.5-vl               # Run with specific model
#   ./run_all_domains.sh --eval_type mmina                # Run mmina domains instead
# ==============================================================================

# Define domains for each evaluation type (pipe-separated to handle spaces in names)
declare -A EVAL_DOMAINS
EVAL_DOMAINS["mmina"]="shopping|wikipedia"
EVAL_DOMAINS["mind2web"]="test_website|test_domain_Info|test_domain_Service"
EVAL_DOMAINS["webvoyager"]="Google Map|Amazon|Allrecipes|Coursera"

# Default values
EVAL_TYPE="webvoyager"
MODEL="qwen2.5-vl"
RESULT_DIR="results"
EXTRA_ARGS=""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --eval_type|--evaluation_type)
            EVAL_TYPE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --result_dir)
            RESULT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "This script runs ALL domains for a given evaluation type."
            echo ""
            echo "Options:"
            echo "  --eval_type TYPE       Evaluation type (mmina, mind2web, webvoyager) [default: webvoyager]"
            echo "  --model MODEL          Model to use (default: qwen2.5-vl)"
            echo "  --result_dir DIR       Base results directory (default: results)"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "All other arguments are passed through to run_agent.sh"
            echo ""
            echo "Domains per evaluation type:"
            echo "  mmina:       shopping, wikipedia"
            echo "  mind2web:    test_website, test_domain_Info, test_domain_Service"
            echo "  webvoyager:  Google Map, Amazon, Allrecipes, Coursera"
            exit 0
            ;;
        *)
            # Collect extra arguments to pass to run_agent.sh
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Use default if not specified
if [ -z "$EVAL_TYPE" ]; then
    EVAL_TYPE="webvoyager"
fi

# Validate evaluation type
if [ -z "${EVAL_DOMAINS[$EVAL_TYPE]}" ]; then
    echo "Error: Invalid evaluation type '$EVAL_TYPE'"
    echo "Supported types: mmina, mind2web, webvoyager"
    exit 1
fi

# Create timestamp for this run
DATETIME=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${RESULT_DIR}/${EVAL_TYPE}_all_domains/${MODEL}/${DATETIME}"
mkdir -p "$RUN_DIR"

# Initialize summary file
SUMMARY_FILE="${RUN_DIR}/all_domains_summary.json"
echo "{" > "$SUMMARY_FILE"
echo "  \"eval_type\": \"$EVAL_TYPE\"," >> "$SUMMARY_FILE"
echo "  \"model\": \"$MODEL\"," >> "$SUMMARY_FILE"
echo "  \"datetime\": \"$DATETIME\"," >> "$SUMMARY_FILE"
echo "  \"domains\": {" >> "$SUMMARY_FILE"

# Get domains for this eval type (pipe-separated)
DOMAINS="${EVAL_DOMAINS[$EVAL_TYPE]}"

# Count domains
DOMAIN_TOTAL_COUNT=$(echo "$DOMAINS" | tr '|' '\n' | wc -l)

echo "============================================"
echo "GUI-Agent All-Domains Runner"
echo "============================================"
echo "Evaluation Type: $EVAL_TYPE"
echo "Model:           $MODEL"
echo "Domains:         $(echo "$DOMAINS" | tr '|' ', ')"
echo "Output Dir:      $RUN_DIR"
echo "Extra Args:      $EXTRA_ARGS"
echo "============================================"
echo ""

# Track overall statistics
TOTAL_TASKS=0
TOTAL_SUCCESSFUL=0
DOMAIN_COUNT=0
FIRST_DOMAIN=true

# Process each domain (using IFS to handle pipe-separated values with spaces)
IFS='|' read -ra DOMAIN_ARRAY <<< "$DOMAINS"
for DOMAIN in "${DOMAIN_ARRAY[@]}"; do
    DOMAIN_COUNT=$((DOMAIN_COUNT + 1))

    echo ""
    echo "============================================"
    echo "Processing Domain: $DOMAIN ($DOMAIN_COUNT of $DOMAIN_TOTAL_COUNT)"
    echo "============================================"

    # Run the agent for this domain
    ./run_agent.sh \
        --eval_type "$EVAL_TYPE" \
        --domain "$DOMAIN" \
        --model "$MODEL" \
        --result_dir "$RUN_DIR" \
        $EXTRA_ARGS

    EXIT_CODE=$?

    # Find the score_summary.json for this run
    # The result directory structure is: $RUN_DIR/$EVAL_TYPE/$DOMAIN/$MODEL/$DATETIME/score_summary.json
    DOMAIN_RESULT_DIR=$(find "$RUN_DIR" -path "*/$DOMAIN/*" -name "score_summary.json" -type f 2>/dev/null | head -1)

    if [ -n "$DOMAIN_RESULT_DIR" ]; then
        # Extract metrics from score_summary.json
        DOMAIN_TOTAL=$(jq -r '.total_tasks // 0' "$DOMAIN_RESULT_DIR")
        DOMAIN_SUCCESS=$(jq -r '.successful_tasks // 0' "$DOMAIN_RESULT_DIR")
        DOMAIN_RATE=$(jq -r '.success_rate // 0' "$DOMAIN_RESULT_DIR")

        # Update totals
        TOTAL_TASKS=$((TOTAL_TASKS + DOMAIN_TOTAL))
        TOTAL_SUCCESSFUL=$((TOTAL_SUCCESSFUL + DOMAIN_SUCCESS))

        # Add to summary JSON
        if [ "$FIRST_DOMAIN" = false ]; then
            echo "," >> "$SUMMARY_FILE"
        fi
        FIRST_DOMAIN=false

        echo "    \"$DOMAIN\": {" >> "$SUMMARY_FILE"
        echo "      \"total_tasks\": $DOMAIN_TOTAL," >> "$SUMMARY_FILE"
        echo "      \"successful_tasks\": $DOMAIN_SUCCESS," >> "$SUMMARY_FILE"
        echo "      \"success_rate\": $DOMAIN_RATE," >> "$SUMMARY_FILE"
        echo "      \"result_dir\": \"$(dirname "$DOMAIN_RESULT_DIR")\"" >> "$SUMMARY_FILE"
        echo -n "    }" >> "$SUMMARY_FILE"

        echo "Domain $DOMAIN: $DOMAIN_SUCCESS/$DOMAIN_TOTAL (${DOMAIN_RATE}%)"
    else
        echo "Warning: No score_summary.json found for domain $DOMAIN"

        if [ "$FIRST_DOMAIN" = false ]; then
            echo "," >> "$SUMMARY_FILE"
        fi
        FIRST_DOMAIN=false

        echo "    \"$DOMAIN\": {" >> "$SUMMARY_FILE"
        echo "      \"error\": \"No results found\"," >> "$SUMMARY_FILE"
        echo "      \"exit_code\": $EXIT_CODE" >> "$SUMMARY_FILE"
        echo -n "    }" >> "$SUMMARY_FILE"
    fi
done

# Calculate overall success rate
if [ $TOTAL_TASKS -gt 0 ]; then
    OVERALL_RATE=$(echo "scale=2; $TOTAL_SUCCESSFUL * 100 / $TOTAL_TASKS" | bc)
else
    OVERALL_RATE="0.00"
fi

# Complete summary JSON
echo "" >> "$SUMMARY_FILE"
echo "  }," >> "$SUMMARY_FILE"
echo "  \"overall\": {" >> "$SUMMARY_FILE"
echo "    \"total_tasks\": $TOTAL_TASKS," >> "$SUMMARY_FILE"
echo "    \"successful_tasks\": $TOTAL_SUCCESSFUL," >> "$SUMMARY_FILE"
echo "    \"success_rate\": $OVERALL_RATE" >> "$SUMMARY_FILE"
echo "  }" >> "$SUMMARY_FILE"
echo "}" >> "$SUMMARY_FILE"

# Print final summary
echo ""
echo "============================================"
echo "ALL DOMAINS SUMMARY"
echo "============================================"
echo "Evaluation Type:   $EVAL_TYPE"
echo "Model:             $MODEL"
echo "Domains Processed: $DOMAIN_COUNT"
echo "--------------------------------------------"
echo "Total Tasks:       $TOTAL_TASKS"
echo "Successful:        $TOTAL_SUCCESSFUL"
echo "Overall Rate:      ${OVERALL_RATE}%"
echo "============================================"
echo ""
echo "Per-domain results:"
jq -r '.domains | to_entries[] | "  \(.key): \(.value.success_rate // "N/A")%"' "$SUMMARY_FILE" 2>/dev/null || echo "  (see $SUMMARY_FILE for details)"
echo ""
echo "Summary saved to: $SUMMARY_FILE"
echo "============================================"
