#include "gen_pair.h"

void trim_string(char *str)
{
    int len = strlen(str);
    while (len > 0 && isspace((unsigned char)str[len - 1]))
    {
        str[len - 1] = '\0';
        len--;
    }
}

int parse_filename(const char *filepath, FileEntry *entry)
{
    const char *basename = strrchr(filepath, '/');
    if (!basename)
    {
        basename = filepath; // 如果没有找到 '/', 那么整个字符串就是文件名
    }
    else
    {
        basename++; // 移过 '/'
    }

    // 解析年、日、时（HHMM格式）
    int year, day, hourMinute;
    int extracted = sscanf(basename, "%*[^.].%d.%d.%d.%*s", &year, &day, &hourMinute);
    if (extracted == 3)
    {
        // 使用 encode_time 函数将年、日、时合并成一个整数
        entry->time.year = year;
        entry->time.jday = day;
        entry->time.hourminute = hourMinute;
        strncpy(entry->filename, filepath, 255); // 保存完整文件路径
        entry->filename[255] = '\0';             // 确保字符串以空字符结尾
        return 1;                                // 成功解析
    }
    return 0; // 解析失败
}

int compare_entries(const void *a, const void *b)
{
    const FileEntry *entry1 = (const FileEntry *)a;
    const FileEntry *entry2 = (const FileEntry *)b;
    if (entry1->time.year != entry2->time.year)
    {
        return (entry1->time.year > entry2->time.year) - (entry1->time.year < entry2->time.year);
    }
    if (entry1->time.jday != entry2->time.jday)
    {
        return (entry1->time.jday > entry2->time.jday) - (entry1->time.jday < entry2->time.jday);
    }
    if (entry1->time.hourminute != entry2->time.hourminute)
    {
        return (entry1->time.hourminute > entry2->time.hourminute) - (entry1->time.hourminute < entry2->time.hourminute);
    }
    return 0;
}

void process_file_list(const char *file_name, FileEntry **entries, size_t *count)
{
    FILE *file = fopen(file_name, "r");
    char line[MAX_LINE_LENGTH];
    *count = 0;
    *entries = malloc(1000 * sizeof(FileEntry)); // Assuming a maximum of 1000 entries

    if (!file)
    {
        fprintf(stderr, "Unable to open file %s\n", file_name);
        return;
    }

    while (fgets(line, sizeof(line), file) && *count < 1000)
    {
        trim_string(line); // Apply trim_string to remove any trailing whitespace or newline
        if (parse_filename(line, &(*entries)[*count]))
        {
            (*count)++;
        }
    }
    fclose(file);
}

void find_matching_files(const char *filelist1, const char *filelist2, FilePair **matches, size_t *match_count)
{
    FileEntry *entries1 = NULL, *entries2 = NULL;
    size_t count1 = 0, count2 = 0;

    process_file_list(filelist1, &entries1, &count1);
    process_file_list(filelist2, &entries2, &count2);

    qsort(entries1, count1, sizeof(FileEntry), compare_entries);
    qsort(entries2, count2, sizeof(FileEntry), compare_entries);

    *match_count = 0;
    size_t max_matches = (count1 < count2 ? count1 : count2);
    *matches = malloc(max_matches * sizeof(FilePair));

    size_t i = 0, j = 0;
    while (i < count1 && j < count2)
    {
        int cmp = compare_entries(&entries1[i], &entries2[j]);
        if (cmp == 0)
        {
            strcpy((*matches)[*match_count].source_path, entries1[i].filename);
            strcpy((*matches)[*match_count].station_path, entries2[j].filename);
            (*matches)[*match_count].time.year = entries1[i].time.year;
            (*matches)[*match_count].time.jday = entries1[i].time.jday;
            (*matches)[*match_count].time.hourminute = entries1[i].time.hourminute;
            (*matches)[*match_count].index = *match_count; // 设置索引
            (*match_count)++;
            i++;
            j++;
        }
        else if (cmp < 0)
        {
            i++;
        }
        else
        {
            j++;
        }
    }

    free(entries1);
    free(entries2);
}
