import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashSet;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class LowFrequencyFilter {
    public static class FilterMapper extends Mapper<Object, Text, Text, IntWritable> {
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {
            // Đọc dòng dữ liệu từ input
            String line = value.toString();
            String[] parts = line.split("\\s+");

            // Sử dụng termId làm key và frequency là value
            context.write(
                    new Text(parts[0]),
                    new IntWritable(Integer.parseInt(parts[2]))
            );
        }
    }

    public static class IntSumCombiner extends Reducer<Text, IntWritable, Text, IntWritable> {
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            // Tính tổng frequency của termid từ tất cả các văn bản
            int totalFrequency = 0;
            for (IntWritable value : values) {
                totalFrequency += value.get();
            }

            context.write(key, new IntWritable(totalFrequency));
        }
    }

    public static class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {

            // Tính tổng frequency của termid từ tất cả các văn bản
            int totalFrequency = 0;
            for (IntWritable value : values) {
                totalFrequency += value.get();
            }

            // Kiểm tra tổng frequency của termid, nếu >= 3 thì ghi vào output
            if (totalFrequency >= 3) {
                context.write(key, new IntWritable(totalFrequency));
            }
        }
    }

    public static class TokenizerMapper extends Mapper<Object, Text, Text, Text> {
        HashSet<String> Terms = new HashSet<>();

        @Override
        protected void setup(Context context) throws IOException {
            Configuration conf = context.getConfiguration();
            FileSystem fs = FileSystem.get(conf);
            String path = conf.get("filteredFile");

            // get the filtered terms from previous file
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(new Path(path))))) {
                String term;
                while ((term = reader.readLine()) != null) {
                    String[] part = term.split("\\s+");
                    Terms.add(part[0]);
                }
            }
        }

        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {
            String line = value.toString();
            String[] parts = line.split("\\s+");

            String termId = parts[0];

            // Kiểm tra xem termId có nằm trong danh sách đã lọc từ job1 không
            if (Terms.contains(termId)) {
                context.write(
                        new Text(termId),
                        new Text(parts[1] + " " + parts[2])
                ); // giữ lại dòng dữ liệu có termId trong danh sách đã lọc
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        conf.set("filteredFile", args[1] + ".tmp/part-r-00000");

        Job job = Job.getInstance(conf, "filter");
        job.setJarByClass(LowFrequencyFilter.class);

        job.setMapperClass(FilterMapper.class);
        job.setCombinerClass(IntSumCombiner.class);
        job.setReducerClass(IntSumReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1] + ".tmp"));
        job.waitForCompletion(true);

        Job job2 = Job.getInstance(conf, "combine");
        job2.setJarByClass(LowFrequencyFilter.class);
        job2.setMapperClass(TokenizerMapper.class);

        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job2, new Path(args[0]));
        FileOutputFormat.setOutputPath(job2, new Path(args[1]));

        System.exit(job2.waitForCompletion(true) ? 0 : 1);
    }
}