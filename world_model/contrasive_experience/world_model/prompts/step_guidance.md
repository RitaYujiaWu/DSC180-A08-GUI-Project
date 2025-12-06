# Step Guidance Prompt

Provide brief, actionable step guidance (under 150 words).

## Context

**Task:** {task}
**Current Step:** {step_num}
**Actions Taken So Far:** {action_history}
**Current State:** {state_description}

## Similar Successful Trajectories at This Step

{success_at_step}

## Common Failures at This Step

{failures_at_step}

## Output Format

1. **Progress Assessment** (1 sentence)
   - Are we on track based on similar successful trajectories?
   - Typical completion is around {avg_steps} steps

2. **Next Action Recommendation** (1-2 sentences)
   - What action type should be considered next?
   - What element or target should be focused on?

3. **Pitfall Warning** (1 sentence)
   - What common mistake should be avoided at this stage?

4. **Verification Reminder** (1 sentence)
   - What should be verified before proceeding?

## Guidelines

- Be specific and actionable
- Reference the current step number
- Consider what successful agents do at this stage
- Warn about common mistakes at similar steps
