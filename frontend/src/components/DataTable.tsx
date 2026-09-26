import React from 'react';

export interface Column<T> {
  key: string;
  header: string;
  render?: (item: T) => React.ReactNode;
  align?: 'left' | 'center' | 'right';
  width?: string;
  isMono?: boolean;
}

interface DataTableProps<T> {
  columns: Column<T>[];
  data: T[];
  keyExtractor: (item: T) => string;
  onRowClick?: (item: T) => void;
  emptyMessage?: string;
  className?: string;
}

export function DataTable<T>({
  columns,
  data,
  keyExtractor,
  onRowClick,
  emptyMessage = 'No records found',
  className = '',
}: DataTableProps<T>) {
  if (data.length === 0) {
    return (
      <div className="p-8 text-center text-xs text-neutral-400 font-mono border border-dashed border-[#1e2430] rounded-lg bg-[#11141a]/40">
        {emptyMessage}
      </div>
    );
  }

  return (
    <div className={`overflow-x-auto rounded-lg border border-[#1e2430] bg-[#11141a] ${className}`}>
      <table className="w-full text-left text-xs border-collapse">
        <thead>
          <tr className="border-b border-[#1e2430] bg-[#161a22]/70 text-neutral-400 font-mono uppercase tracking-wider text-[11px]">
            {columns.map((col) => (
              <th
                key={col.key}
                className={`py-3 px-4 font-medium ${
                  col.align === 'right'
                    ? 'text-right'
                    : col.align === 'center'
                    ? 'text-center'
                    : 'text-left'
                }`}
                style={col.width ? { width: col.width } : undefined}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-[#1e2430]/60">
          {data.map((item) => {
            const key = keyExtractor(item);
            return (
              <tr
                key={key}
                onClick={() => onRowClick && onRowClick(item)}
                className={`transition-colors ${
                  onRowClick ? 'cursor-pointer hover:bg-[#181d26]' : 'hover:bg-[#141820]'
                }`}
              >
                {columns.map((col) => {
                  const content = col.render
                    ? col.render(item)
                    : (item as Record<string, unknown>)[col.key] as React.ReactNode;

                  return (
                    <td
                      key={col.key}
                      className={`py-3 px-4 ${
                        col.isMono ? 'font-mono' : 'font-sans'
                      } text-neutral-200 ${
                        col.align === 'right'
                          ? 'text-right'
                          : col.align === 'center'
                          ? 'text-center'
                          : 'text-left'
                      }`}
                    >
                      {content}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
