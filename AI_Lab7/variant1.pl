% Create an object for every month
days_in_month(01, 31).
days_in_month(02, 28).
days_in_month(03, 31).
days_in_month(04, 30).
days_in_month(05, 31).
days_in_month(06, 30).
days_in_month(07, 31).
days_in_month(08, 31).
days_in_month(09, 30).
days_in_month(10, 31).
days_in_month(11, 30).
days_in_month(12, 31).

% Add days
add_date(Date, DaysToAdd, NewDate) :-
    % Extract the day and month from the input
    sub_string(Date, 0, 2, _, DayStr),
    sub_string(Date, 2, 2, _, MonthStr),
    % Turn them into numbers
    atom_number(DayStr, Day),
    atom_number(MonthStr, Month),
    % Add number of dates
    days_in_month(Month, DaysInMonth),
    (   DaysToAdd + Day =< DaysInMonth  /* Don't have to go to next month */
    ->  NewDay is Day + DaysToAdd,
        format_date(NewDay, Month, NewDate)
    ;   RemainingDays is DaysToAdd - (DaysInMonth - Day + 1), /* Has to go to next month */
        (   Month = 12
        ->  NextMonth is 1
        ;   NextMonth is Month + 1
        ),
        format_date(1, NextMonth, FirstDayOfNextMonth),
        add_date(FirstDayOfNextMonth, RemainingDays, NewDate)
    ).



% Subtract days
sub_date(Date, DaysToSubtract, NewDate) :-
    % Extract the day and month from the input
    sub_string(Date, 0, 2, _, DayStr),
    sub_string(Date, 2, 2, _, MonthStr),
    % Turn them into numbers
    atom_number(DayStr, Day),
    atom_number(MonthStr, Month),
    % Subtract number of dates
    (   Day - DaysToSubtract >= 0    /* Don't have to go to the previous month */
    ->  NewDay is Day - DaysToSubtract,
        format_date(NewDay, Month, NewDate)
    ;   RemainingDays is DaysToSubtract - Day,  /* Has to go to the previous month */
        (   Month = 1
        ->  PrevMonth is 12
        ;   PrevMonth is Month - 1
        ),
        days_in_month(PrevMonth, DaysInPrevMonth),
        NewDay is DaysInPrevMonth - RemainingDays + 1,  /* +1 because we already counted the first day of the month */
        format_date(NewDay, PrevMonth, NewDate)
    ).



% Formatted Output
format_date(Day, Month, FormattedDate) :-
    format(atom(DayStr), '~|~`0t~d~2+', Day),
    format(atom(MonthStr), '~|~`0t~d~2+', Month),
    atom_concat(DayStr, MonthStr, FormattedDate).

