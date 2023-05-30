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


add_date(Date, DaysToAdd, Result) :-
    sub_string(Date, 0, 2, _, DayStr),
    sub_string(Date, 2, 2, _, MonthStr),
    atom_number(DayStr, Day),
    atom_number(MonthStr, Month),
    days_in_month(Month, DaysInMonth),
    (   DaysToAdd + Day =< DaysInMonth
    ->  NewDay is Day + DaysToAdd,
        format_date(NewDay, Month, Result)
    ;   RemainingDays is DaysToAdd - (DaysInMonth - Day + 1),
        (   Month = 12
        ->  NextMonth is 1
        ;   NextMonth is Month + 1
        ),
        format_date(1, NextMonth, NewDate),
        add_date(NewDate, RemainingDays, NewResult),
        Result = NewResult
    ).

sub_date(Date, DaysToSubtract, Result) :-
    sub_string(Date, 0, 2, _, DayStr),
    sub_string(Date, 2, 2, _, MonthStr),
    atom_number(DayStr, Day),
    atom_number(MonthStr, Month),
    (   Day - DaysToSubtract > 0
    ->  NewDay is Day - DaysToSubtract,
        format_date(NewDay, Month, Result)
    ;   RemainingDays is DaysToSubtract - Day,
        (   Month = 1
        ->  PrevMonth is 12
        ;   PrevMonth is Month - 1
        ),
        days_in_month(PrevMonth, DaysInPrevMonth),
        NewDay is DaysInPrevMonth - RemainingDays,
        format_date(NewDay, PrevMonth, Result)
    ).

format_date(Day, Month, DateStr) :-
    (Day < 10 -> atom_concat('0', Day, DayStr); atom_number(DayStr, Day)),
    (Month < 10 -> atom_concat('0', Month, MonthStr); atom_number(MonthStr, 
Month)),
    atom_concat(DayStr, MonthStr, DateStr).

