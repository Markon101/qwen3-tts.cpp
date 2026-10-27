"""Associate private speech-code dumps only when their provenance is unambiguous.

Count and contiguous-name checks are valid only for a fresh, exclusive dump
directory used by one serial process. Callers must preserve those conditions.
"""


def associate_code_dumps(responses, filenames):
    """Return response-aligned filenames and whether association is ambiguous.

    A failed response may have generated a dump before a later synthesis stage
    failed. Refuse every association unless the actual filenames exactly match
    codes-1.i32 through codes-N.i32 for N successful responses. Failed responses
    always map to None; an empty response/dump set is unambiguous.
    """
    responses = list(responses)
    filenames = list(filenames)
    successes = sum(bool(response.get('success')) for response in responses)
    expected = [f'codes-{index}.i32' for index in range(1, successes + 1)]
    if len(filenames) != successes or set(filenames) != set(expected):
        return [None] * len(responses), True

    names = iter(expected)
    return [next(names) if response.get('success') else None
            for response in responses], False
