# Contrastive Summary Prompt

You are an expert at analyzing GUI automation trajectories.

## Your Goal

Generate a concise contrastive summary that helps an agent succeed at the current task by learning from similar past attempts.

## Input

**Current Task:** {task}
**Domain:** {domain}

**Successful Trajectories:**
{success_summaries}

**Failed Trajectories:**
{failure_summaries}

## Output Format

Generate a structured analysis with:

1. **SUCCESS PATTERNS** (2-3 bullet points):
   - Identify specific actions or sequences that led to successful completion
   - Note timing, order, or approach that worked well

2. **COMMON MISTAKES** (2-3 bullet points):
   - Identify specific actions or approaches that led to failure
   - Note what to avoid or watch out for

3. **KEY DIVERGENCE** (1 sentence):
   - Identify where the paths of success and failure typically diverge
   - This is the critical decision point

4. **RECOMMENDATION** (1-2 sentences):
   - Provide specific, actionable guidance for the current task
   - Reference the patterns identified above

## Guidelines

- Keep output under 200 words
- Focus on actionable, specific insights
- Avoid generic advice
- Reference actual actions when possible (click, type, scroll, etc.)
- Consider the order and timing of actions
