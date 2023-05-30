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
add_date(Date, DaysToAdd) :-
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
        format_date(NewDay, Month)
    ;   RemainingDays is DaysToAdd - (DaysInMonth - Day + 1), /* Has to go to next month */
        (   Month = 12
        ->  NextMonth is 1
        ;   NextMonth is Month + 1
        ),
        format_date(1, NextMonth),
        add_date("0106", RemainingDays)
    ).

% Subtract days
sub_date(Date, DaysToSubtract) :-
    % Extract the day and month from the input
    sub_string(Date, 0, 2, _, DayStr),
    sub_string(Date, 2, 2, _, MonthStr),
    % Turn them into numbers
    atom_number(DayStr, Day),
    atom_number(MonthStr, Month),
    % Subtract number of dates
    (   Day - DaysToSubtract > 0    /* Don't have to go to the previous month */
    ->  NewDay is Day - DaysToSubtract,
        format_date(NewDay, Month)
    ;   RemainingDays is DaysToSubtract - Day,  /* Has to go to the previous month */
        (   Month = 1
        ->  PrevMonth is 12
        ;   PrevMonth is Month - 1
        ),
        days_in_month(PrevMonth, DaysInPrevMonth),
        NewDay is DaysInPrevMonth - RemainingDays,
        format_date(NewDay, PrevMonth)
    ).

% Formatted Output
format_date(Day, Month) :-
    (Day < 10 -> atom_concat('0', Day, DayStr); atom_number(DayStr, Day)),
    (Month < 10 -> atom_concat('0', Month, MonthStr); atom_number(MonthStr, Month)),
    atom_concat(DayStr, MonthStr, DateStr),
    format('"~w"',[DateStr]).
